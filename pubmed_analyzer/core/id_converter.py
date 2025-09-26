import asyncio
import xml.etree.ElementTree as ET
import logging
from typing import List, Optional
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

from ..models.paper import Paper
from ..utils.ncbi_client import NCBIClient

logger = logging.getLogger(__name__)


class PMIDToPMCConverter:
    """Convert between PMID and PMC IDs and check OA availability"""

    def __init__(self, email: str, api_key: Optional[str] = None):
        self.email = email
        self.api_key = api_key
        self._conversion_cache = {}  # Simple in-memory cache

    async def enrich_with_pmcids(self, papers: List[Paper], batch_size: int = None) -> None:
        """
        FAST bulk PMC ID conversion using optimized NCBI ID Converter API

        Args:
            papers: List of Paper objects to enrich
            batch_size: Number of papers to process in each batch (auto-optimized if None)
        """
        if not papers:
            return

        # Use much larger batch sizes for the optimized approach
        if batch_size is None:
            if self.api_key:
                batch_size = min(200, max(100, len(papers)))  # Much larger batches with API key
            else:
                batch_size = min(100, max(50, len(papers)))   # Still large batches without API key

        logger.info(f"🚀 FAST PMC conversion: {len(papers)} papers in batches of {batch_size}")

        async with NCBIClient(self.email, self.api_key) as client:
            with tqdm(total=len(papers), desc="Converting PMIDs to PMC IDs", unit="paper") as pbar:
                for i in range(0, len(papers), batch_size):
                    batch = papers[i : i + batch_size]

                    await self._bulk_convert_batch(client, batch)
                    pbar.update(len(batch))

                    # Minimal delay between batches
                    if i + batch_size < len(papers):
                        delay = 0.2 if self.api_key else 0.5  # Much shorter delays
                        await asyncio.sleep(delay)

    async def _bulk_convert_batch(self, client: NCBIClient, papers: List[Paper]) -> None:
        """Bulk convert a batch of papers using NCBI ID Converter API"""
        try:
            # Check cache first and extract uncached PMIDs
            pmids_to_convert = []
            for paper in papers:
                if paper.clean_pmid:
                    if paper.clean_pmid in self._conversion_cache:
                        # Use cached result
                        pmcid = self._conversion_cache[paper.clean_pmid]
                        if pmcid:
                            paper.pmcid = pmcid
                            paper.has_fulltext = True
                    else:
                        pmids_to_convert.append(paper.clean_pmid)

            if not pmids_to_convert:
                return  # All were cached

            # Use NCBI ID Converter API for bulk conversion (much faster)
            pmid_to_pmcid = await self._bulk_pmid_to_pmcid_conversion(client, pmids_to_convert)

            # Apply conversions to papers and update cache
            for paper in papers:
                if paper.clean_pmid in pmid_to_pmcid:
                    pmcid = pmid_to_pmcid[paper.clean_pmid]
                    # Cache the result (including None for no conversion)
                    self._conversion_cache[paper.clean_pmid] = pmcid

                    if pmcid:
                        paper.pmcid = pmcid
                        paper.has_fulltext = True  # Assume PMC papers have full text
                elif paper.clean_pmid in pmids_to_convert:
                    # Cache negative result
                    self._conversion_cache[paper.clean_pmid] = None

        except Exception as e:
            logger.error(f"Bulk conversion failed, falling back to individual conversion: {e}")
            # Fallback to old method for this batch only
            await self._process_batch_fallback(client, papers)

    async def _bulk_pmid_to_pmcid_conversion(self, client: NCBIClient, pmids: List[str]) -> dict:
        """
        Use NCBI ID Converter API for bulk PMID to PMC conversion
        This is MUCH faster than individual elink calls
        """
        if not pmids:
            return {}

        try:
            # Join PMIDs for bulk query
            pmid_string = ",".join(pmids)

            # Use the ID Converter API endpoint - much faster for bulk operations
            url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
            params = {
                'tool': 'PubMedAnalyzer',
                'email': self.email,
                'ids': pmid_string,
                'format': 'json',
                'versions': 'no'
            }

            if self.api_key:
                params['api_key'] = self.api_key

            response = await client.session.get(url, params=params)

            if response.status == 200:
                data = await response.json()

                # Parse the conversion results
                pmid_to_pmcid = {}

                if 'records' in data:
                    for record in data['records']:
                        pmid = record.get('pmid')
                        pmcid = record.get('pmcid')

                        if pmid and pmcid:
                            # Convert PMID to string for consistent key matching
                            pmid_str = str(pmid)

                            # Clean up PMC ID format
                            if pmcid.startswith('PMC'):
                                pmid_to_pmcid[pmid_str] = pmcid
                            else:
                                pmid_to_pmcid[pmid_str] = f'PMC{pmcid}'

                logger.debug(f"Bulk converted {len(pmid_to_pmcid)}/{len(pmids)} PMIDs to PMC IDs")
                return pmid_to_pmcid

            else:
                logger.warning(f"ID Converter API returned status {response.status}")
                return {}

        except Exception as e:
            logger.error(f"Bulk PMID to PMC conversion failed: {e}")
            return {}

    async def _process_batch_fallback(self, client: NCBIClient, papers: List[Paper]) -> None:
        """Fallback to the old sequential method if bulk conversion fails"""
        for paper in papers:
            try:
                if not paper.pmcid:
                    paper.pmcid = await self._convert_pmid_to_pmcid(client, paper.clean_pmid)

                if paper.pmcid:
                    paper.has_fulltext = True

                # Very small delay for fallback
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Failed to enrich paper {paper.pmid}: {e}")
                paper.error_message = str(e)

    async def _process_batch(self, client: NCBIClient, papers: List[Paper]) -> None:
        """Process a batch of papers for PMC ID conversion with sequential processing"""
        # Process sequentially instead of concurrently to avoid rate limit storms
        for i, paper in enumerate(papers):
            try:
                await self._enrich_single_paper(client, paper)

                # Small delay between individual papers
                if i < len(papers) - 1:
                    await asyncio.sleep(0.3)

            except Exception as e:
                logger.error(f"Failed to enrich paper {paper.pmid}: {e}")
                paper.error_message = str(e)

    async def _enrich_single_paper(self, client: NCBIClient, paper: Paper) -> None:
        """
        Enrich a single paper with PMC ID and OA information

        Args:
            client: NCBI client
            paper: Paper object to enrich
        """
        try:
            # Convert PMID to PMC ID
            if not paper.pmcid:
                paper.pmcid = await self._convert_pmid_to_pmcid(client, paper.clean_pmid)

            # Check OA availability if we have a PMC ID
            if paper.pmcid:
                oa_info = await self._check_oa_availability(client, paper.pmcid)
                if oa_info:
                    paper.pmc_metadata = oa_info
                    paper.has_fulltext = True
                    paper.license = oa_info.get('license', 'Unknown')
                    paper.is_retracted = oa_info.get('retracted', 'N/A').lower() == 'yes'

        except Exception as e:
            logger.error(f"Failed to enrich paper {paper.pmid}: {e}")
            paper.error_message = str(e)

    async def _convert_pmid_to_pmcid(self, client: NCBIClient, pmid: str) -> Optional[str]:
        """
        Convert PMID to PMC ID using elink

        Args:
            client: NCBI client
            pmid: Clean PMID (without PMID: prefix)

        Returns:
            PMC ID if available, None otherwise
        """
        try:
            # Method 1: Use elink (most reliable)
            pmcid = await client.convert_pmid_to_pmcid(pmid)
            if pmcid:
                logger.debug(f"Converted PMID {pmid} -> {pmcid} via elink")
                return pmcid

            # Method 2: Cross-search PMC database
            pmcid = await self._cross_search_pmc(client, pmid)
            if pmcid:
                logger.debug(f"Converted PMID {pmid} -> {pmcid} via cross-search")
                return pmcid

            logger.debug(f"No PMC ID found for PMID {pmid}")
            return None

        except Exception as e:
            logger.error(f"Failed to convert PMID {pmid} to PMC ID: {e}")
            return None

    async def _cross_search_pmc(self, client: NCBIClient, pmid: str) -> Optional[str]:
        """
        Cross-search PMC database for PMID

        Args:
            client: NCBI client
            pmid: Clean PMID

        Returns:
            PMC ID if found, None otherwise
        """
        try:
            params = {
                'db': 'pmc',
                'term': f'{pmid}[PMID]',
                'retmax': '1',
                'retmode': 'json'
            }

            response = await client.make_request('esearch.fcgi', params)
            data = await response.json()

            search_result = data.get('esearchresult', {})
            ids = search_result.get('idlist', [])

            if ids:
                return f"PMC{ids[0]}"

            return None

        except Exception as e:
            logger.error(f"Cross-search failed for PMID {pmid}: {e}")
            return None

    async def _check_oa_availability(self, client: NCBIClient, pmcid: str) -> Optional[dict]:
        """
        Check Open Access availability using PMC OA service

        Args:
            client: NCBI client
            pmcid: PMC ID (with or without PMC prefix)

        Returns:
            OA metadata if available, None otherwise
        """
        try:
            # Ensure PMC prefix
            clean_pmcid = pmcid.replace('PMC', '')
            full_pmcid = f"PMC{clean_pmcid}"

            params = {'id': full_pmcid}
            response = await client.make_pmc_request(params)
            xml_content = await response.text()

            # Parse OA response
            root = ET.fromstring(xml_content)
            record = root.find('records/record')

            if record is None:
                logger.debug(f"No OA record found for {full_pmcid}")
                return None

            # Extract metadata
            metadata = {
                'pmcid': full_pmcid,
                'citation': record.attrib.get('citation', 'N/A'),
                'license': record.attrib.get('license', 'N/A'),
                'retracted': record.attrib.get('retracted', 'N/A'),
            }

            # Find PDF URL
            pdf_url = None
            for link in record.findall('link'):
                if link.attrib.get('format') == 'pdf':
                    pdf_url = link.attrib.get('href')
                    break

            if pdf_url:
                metadata['pdf_url'] = pdf_url

            logger.debug(f"Found OA metadata for {full_pmcid}: {metadata}")
            return metadata

        except Exception as e:
            logger.error(f"Failed to check OA availability for {pmcid}: {e}")
            return None