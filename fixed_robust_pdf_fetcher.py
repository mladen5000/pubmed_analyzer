#!/usr/bin/env python3
"""
FIXED Robust PDF Fetcher - Addresses all 0% success rate issues
"""

import asyncio
import aiohttp
import xml.etree.ElementTree as ET
import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import time

from pubmed_analyzer.models.paper import Paper
from pubmed_analyzer.utils.validators import PDFValidator

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Result of a PDF download attempt"""
    pmid: str
    success: bool
    file_path: Optional[str] = None
    strategy_used: Optional[str] = None
    error_message: Optional[str] = None
    file_size: int = 0


class FixedPDFDownloadStrategy(ABC):
    """Fixed abstract base class for PDF download strategies"""

    @abstractmethod
    async def can_handle(self, paper: Paper) -> bool:
        """Check if this strategy can handle downloading this paper"""
        pass

    @abstractmethod
    async def download(self, session: aiohttp.ClientSession, paper: Paper, pdf_dir: str) -> bool:
        """Download the PDF for this paper"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging"""
        pass

    @property
    @abstractmethod
    def priority(self) -> int:
        """Strategy priority (lower = higher priority)"""
        pass


class FixedPMCOAServiceStrategy(FixedPDFDownloadStrategy):
    """FIXED: PMC Open Access service with correct API usage"""

    @property
    def name(self) -> str:
        return "Fixed PMC OA Service"

    @property
    def priority(self) -> int:
        return 1

    async def can_handle(self, paper: Paper) -> bool:
        """Can handle if paper has PMC ID"""
        return paper.pmcid is not None

    async def download(self, session: aiohttp.ClientSession, paper: Paper, pdf_dir: str) -> bool:
        """Download using correct PMC OA Service API"""
        if not paper.pmcid:
            return False

        try:
            # Step 1: Get PDF URL from PMC OA Service
            pdf_info = await self._get_pdf_info(session, paper.pmcid)
            if not pdf_info or 'pdf_url' not in pdf_info:
                logger.debug(f"No PDF URL found via OA Service for {paper.pmcid}")
                return False

            pdf_url = pdf_info['pdf_url']
            logger.debug(f"Found PDF URL via OA Service: {pdf_url}")

            # Step 2: Download the PDF
            return await self._download_from_url(session, paper, pdf_url, pdf_dir)

        except Exception as e:
            logger.error(f"Fixed PMC OA Service failed for {paper.pmcid}: {e}")
            return False

    async def _get_pdf_info(self, session: aiohttp.ClientSession, pmc_id: str) -> Optional[Dict]:
        """Get PDF info from PMC OA Service using correct API"""
        try:
            clean_pmc = pmc_id.replace('PMC', '')
            full_pmc = f"PMC{clean_pmc}"

            url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
            params = {'id': full_pmc}

            headers = {
                'User-Agent': 'PubMedAnalyzer/1.0 (mailto:researcher@university.edu)',
                'Accept': 'application/xml, text/xml',
            }

            async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status != 200:
                    return None

                xml_content = await response.text()
                root = ET.fromstring(xml_content)

                # Check for errors
                error = root.find('error')
                if error is not None:
                    return None

                # Find PDF link
                records = root.find('records')
                if records is None:
                    return None

                record = records.find('record')
                if record is None:
                    return None

                # Look for PDF link
                for link in record.findall('link'):
                    if 'pdf' in link.get('format', '').lower():
                        href = link.get('href')
                        if href:
                            # Convert FTP to HTTPS
                            if href.startswith('ftp://ftp.ncbi.nlm.nih.gov'):
                                href = href.replace('ftp://ftp.ncbi.nlm.nih.gov', 'https://ftp.ncbi.nlm.nih.gov')
                            return {'pdf_url': href}

                return None

        except Exception as e:
            logger.error(f"Failed to get PDF info from OA Service for {pmc_id}: {e}")
            return None

    async def _download_from_url(self, session: aiohttp.ClientSession, paper: Paper, pdf_url: str, pdf_dir: str) -> bool:
        """Download PDF from URL"""
        try:
            headers = {
                'User-Agent': 'PubMedAnalyzer/1.0 (mailto:researcher@university.edu)',
                'Accept': 'application/pdf,*/*',
            }

            async with session.get(pdf_url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    content = await response.read()

                    if len(content) < 1024 or not content.startswith(b'%PDF'):
                        return False

                    # Save file
                    pdf_path = Path(pdf_dir) / f"{paper.pmcid}.pdf"
                    pdf_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(pdf_path, 'wb') as f:
                        f.write(content)

                    if PDFValidator.is_valid_pdf(str(pdf_path)):
                        paper.pdf_path = str(pdf_path)
                        paper.download_success = True
                        logger.info(f"✅ Downloaded {paper.pmcid} via Fixed PMC OA Service")
                        return True
                    else:
                        os.unlink(pdf_path)  # Remove invalid file
                        return False

                return False

        except Exception as e:
            logger.error(f"Failed to download from URL {pdf_url}: {e}")
            return False


class EuropePMCStrategy(FixedPDFDownloadStrategy):
    """NEW: EuropePMC direct PDF access - highest success rate!"""

    @property
    def name(self) -> str:
        return "EuropePMC Direct"

    @property
    def priority(self) -> int:
        return 0  # Highest priority - this works best!

    async def can_handle(self, paper: Paper) -> bool:
        """Can handle if paper has PMC ID"""
        return paper.pmcid is not None

    async def download(self, session: aiohttp.ClientSession, paper: Paper, pdf_dir: str) -> bool:
        """Download directly from EuropePMC"""
        if not paper.pmcid:
            return False

        try:
            clean_pmc = paper.pmcid.replace('PMC', '')
            pdf_url = f"https://europepmc.org/articles/PMC{clean_pmc}?pdf=render"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'application/pdf,*/*',
            }

            logger.debug(f"Trying EuropePMC: {pdf_url}")

            async with session.get(pdf_url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '').lower()
                    if 'pdf' in content_type:
                        content = await response.read()

                        if len(content) > 1024 and content.startswith(b'%PDF'):
                            # Save file
                            pdf_path = Path(pdf_dir) / f"{paper.pmcid}.pdf"
                            pdf_path.parent.mkdir(parents=True, exist_ok=True)

                            with open(pdf_path, 'wb') as f:
                                f.write(content)

                            if PDFValidator.is_valid_pdf(str(pdf_path)):
                                paper.pdf_path = str(pdf_path)
                                paper.download_success = True
                                logger.info(f"✅ Downloaded {paper.pmcid} via EuropePMC")
                                return True
                            else:
                                os.unlink(pdf_path)

            return False

        except Exception as e:
            logger.error(f"EuropePMC download failed for {paper.pmcid}: {e}")
            return False


class FixedDirectPMCStrategy(FixedPDFDownloadStrategy):
    """FIXED: Direct PMC access with working URL patterns"""

    @property
    def name(self) -> str:
        return "Fixed Direct PMC"

    @property
    def priority(self) -> int:
        return 2

    async def can_handle(self, paper: Paper) -> bool:
        """Can handle if paper has PMC ID"""
        return paper.pmcid is not None

    async def download(self, session: aiohttp.ClientSession, paper: Paper, pdf_dir: str) -> bool:
        """Try working PMC URL patterns"""
        if not paper.pmcid:
            return False

        clean_pmc = paper.pmcid.replace('PMC', '')

        # Working URL patterns based on our tests
        url_patterns = [
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{clean_pmc}/pdf/",
            f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{clean_pmc}/pdf/",
        ]

        for url in url_patterns:
            if await self._try_download_url(session, url, paper, pdf_dir):
                return True

        return False

    async def _try_download_url(self, session: aiohttp.ClientSession, url: str, paper: Paper, pdf_dir: str) -> bool:
        """Try downloading from a specific URL with proper redirect following"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'application/pdf,text/html,*/*',
            }

            # Allow redirects and follow them
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30), allow_redirects=True) as response:
                if response.status == 200:
                    # Check if final URL looks like a PDF
                    final_url = str(response.url)
                    if '.pdf' in final_url.lower():
                        content = await response.read()

                        if len(content) > 1024 and content.startswith(b'%PDF'):
                            # Save file
                            pdf_path = Path(pdf_dir) / f"{paper.pmcid}.pdf"
                            pdf_path.parent.mkdir(parents=True, exist_ok=True)

                            with open(pdf_path, 'wb') as f:
                                f.write(content)

                            if PDFValidator.is_valid_pdf(str(pdf_path)):
                                paper.pdf_path = str(pdf_path)
                                paper.download_success = True
                                logger.info(f"✅ Downloaded {paper.pmcid} via Fixed Direct PMC")
                                return True
                            else:
                                os.unlink(pdf_path)

            return False

        except Exception as e:
            logger.debug(f"Failed to download from {url}: {e}")
            return False


class ImprovedDOIStrategy(FixedPDFDownloadStrategy):
    """IMPROVED: DOI resolution with better handling"""

    @property
    def name(self) -> str:
        return "Improved DOI"

    @property
    def priority(self) -> int:
        return 3

    async def can_handle(self, paper: Paper) -> bool:
        """Can handle if paper has DOI"""
        return paper.doi is not None

    async def download(self, session: aiohttp.ClientSession, paper: Paper, pdf_dir: str) -> bool:
        """Try DOI resolution with improved handling"""
        if not paper.doi:
            return False

        try:
            # Try DOI resolution with PDF accept header
            doi_url = f"https://doi.org/{paper.doi}"

            headers = {
                'User-Agent': 'PubMedAnalyzer/1.0 (mailto:researcher@university.edu)',
                'Accept': 'application/pdf,*/*',
            }

            async with session.get(doi_url, headers=headers, timeout=aiohttp.ClientTimeout(total=30), allow_redirects=True) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '').lower()
                    final_url = str(response.url).lower()

                    # Check if we got a PDF or if URL suggests PDF access
                    if 'pdf' in content_type or '.pdf' in final_url:
                        content = await response.read()

                        if len(content) > 1024 and content.startswith(b'%PDF'):
                            # Save file
                            pdf_path = Path(pdf_dir) / f"{paper.clean_pmid}.pdf"
                            pdf_path.parent.mkdir(parents=True, exist_ok=True)

                            with open(pdf_path, 'wb') as f:
                                f.write(content)

                            if PDFValidator.is_valid_pdf(str(pdf_path)):
                                paper.pdf_path = str(pdf_path)
                                paper.download_success = True
                                logger.info(f"✅ Downloaded {paper.pmid} via Improved DOI")
                                return True
                            else:
                                os.unlink(pdf_path)

            return False

        except Exception as e:
            logger.error(f"Improved DOI download failed for {paper.doi}: {e}")
            return False


class FixedRobustPDFFetcher:
    """FIXED version of robust PDF fetcher with working strategies"""

    def __init__(self, pdf_dir: str = "pdfs", batch_size: int = 5):
        self.pdf_dir = Path(pdf_dir)
        self.pdf_dir.mkdir(exist_ok=True)
        self.batch_size = batch_size

        # Strategies ordered by success rate (based on our testing)
        self.strategies = [
            EuropePMCStrategy(),        # Highest success rate
            FixedPMCOAServiceStrategy(), # Fixed PMC OA Service
            FixedDirectPMCStrategy(),   # Fixed direct PMC
            ImprovedDOIStrategy(),      # Improved DOI handling
        ]

        logger.info(f"Initialized FixedRobustPDFFetcher with {len(self.strategies)} working strategies")

    async def download_batch(self, papers: List[Paper]) -> Dict[str, Any]:
        """Download PDFs for a batch of papers"""
        start_time = time.time()
        results = []

        # Filter papers that have identifiers we can work with
        downloadable_papers = [p for p in papers if p.pmcid or p.doi]
        if not downloadable_papers:
            logger.warning("No papers with PMC IDs or DOIs found")
            return {
                'success_rate': 0.0,
                'successful_downloads': 0,
                'total_papers': len(papers),
                'results': results
            }

        logger.info(f"Attempting to download PDFs for {len(downloadable_papers)} papers with identifiers")

        async with aiohttp.ClientSession() as session:
            # Process in batches
            for i in range(0, len(downloadable_papers), self.batch_size):
                batch = downloadable_papers[i:i + self.batch_size]

                # Process batch concurrently
                tasks = [self._download_single_paper(session, paper) for paper in batch]
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)

                # Add delay between batches
                if i + self.batch_size < len(downloadable_papers):
                    await asyncio.sleep(1.0)

        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.success)
        success_rate = successful / len(downloadable_papers) if downloadable_papers else 0.0

        logger.info(f"PDF download complete: {successful}/{len(downloadable_papers)} successful ({success_rate:.1%})")

        return {
            'success_rate': success_rate,
            'successful_downloads': successful,
            'total_papers': len(downloadable_papers),
            'total_time': total_time,
            'results': results
        }

    async def _download_single_paper(self, session: aiohttp.ClientSession, paper: Paper) -> DownloadResult:
        """Download PDF for a single paper using available strategies"""
        for strategy in self.strategies:
            try:
                if await strategy.can_handle(paper):
                    logger.debug(f"Trying {strategy.name} for {paper.pmcid or paper.pmid}")

                    if await strategy.download(session, paper, str(self.pdf_dir)):
                        return DownloadResult(
                            pmid=paper.pmid,
                            success=True,
                            file_path=paper.pdf_path,
                            strategy_used=strategy.name
                        )

            except Exception as e:
                logger.error(f"Strategy {strategy.name} failed for {paper.pmcid or paper.pmid}: {e}")

        # All strategies failed
        return DownloadResult(
            pmid=paper.pmid,
            success=False,
            error_message="All strategies failed"
        )


# Test function
async def test_fixed_fetcher():
    """Test the fixed fetcher with known PMC IDs"""
    from pubmed_analyzer.models.paper import Paper

    # Create test papers with known PMC IDs
    test_papers = [
        Paper(pmid="32526867", pmcid="PMC8443998", title="Test Paper 1"),
        Paper(pmid="31247177", pmcid="PMC6557568", title="Test Paper 2", doi="10.1371/journal.pone.0218004"),
        Paper(pmid="32641130", pmcid="PMC7308628", title="Test Paper 3"),
    ]

    fetcher = FixedRobustPDFFetcher("test_pdfs")
    results = await fetcher.download_batch(test_papers)

    print(f"\n📊 Test Results:")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Downloaded: {results['successful_downloads']}/{results['total_papers']}")

    for result in results['results']:
        status = "✅" if result.success else "❌"
        print(f"{status} {result.pmid}: {result.strategy_used or result.error_message}")


if __name__ == "__main__":
    asyncio.run(test_fixed_fetcher())