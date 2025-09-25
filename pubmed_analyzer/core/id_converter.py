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

    async def enrich_with_pmcids(self, papers: List[Paper], batch_size: int = None) -> None:
        """
        Enrich papers with PMC IDs and OA availability information

        Args:
            papers: List of Paper objects to enrich
            batch_size: Number of papers to process in each batch (auto-optimized if None)
        """
        if not papers:
            return

        # Optimize batch size based on API key and total papers
        if batch_size is None:
            if self.api_key:
                batch_size = min(10, max(3, len(papers) // 10))  # Smaller batches with API key
            else:
                batch_size = min(5, max(2, len(papers) // 20))   # Very small batches without API key

        logger.info(f"Processing {len(papers)} papers in batches of {batch_size} for PMC ID conversion")

        async with NCBIClient(self.email, self.api_key) as client:
            # Process papers in small batches with delays and progress tracking
            total_batches = (len(papers) + batch_size - 1) // batch_size

            with tqdm(total=len(papers), desc="Converting PMIDs to PMC IDs", unit="paper") as pbar:
                for i in range(0, len(papers), batch_size):
                    batch = papers[i : i + batch_size]
                    batch_num = i // batch_size + 1

                    logger.debug(f"Processing PMC ID batch {batch_num}/{total_batches}")

                    await self._process_batch(client, batch)
                    pbar.update(len(batch))

                    # Add delay between batches to avoid rate limits
                    if i + batch_size < len(papers):
                        delay = 2.0 if not self.api_key else 1.0
                        logger.debug(f"Waiting {delay}s between PMC ID conversion batches...")
                        await asyncio.sleep(delay)

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