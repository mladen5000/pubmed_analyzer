import asyncio
import aiohttp
import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict

from ..models.paper import Paper
from ..utils.validators import PDFValidator
from .markdown_converter import MarkdownConverter

try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False
    logging.warning("MarkItDown not available - PDF text extraction will be limited")

logger = logging.getLogger(__name__)


class PDFDownloadStrategy(ABC):
    """Abstract base class for PDF download strategies"""

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


class PMCOAServiceStrategy(PDFDownloadStrategy):
    """Download PDFs using PMC Open Access service (your ftp_pubmed.py approach)"""

    @property
    def name(self) -> str:
        return "PMC OA Service"

    async def can_handle(self, paper: Paper) -> bool:
        """Check if paper has PMC OA metadata with PDF URL"""
        return (
            paper.pmc_metadata is not None
            and 'pdf_url' in paper.pmc_metadata
            and paper.pmc_metadata['pdf_url'] is not None
        )

    async def download(self, session: aiohttp.ClientSession, paper: Paper, pdf_dir: str) -> bool:
        """Download PDF using PMC OA service URL"""
        try:
            pdf_url = paper.pmc_metadata['pdf_url']

            # Convert FTP URLs to HTTP URLs for NCBI PMC
            if pdf_url.startswith('ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/'):
                # Extract the path after /pub/pmc/
                ftp_path = pdf_url.replace('ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/', '')
                # Convert to HTTP equivalent
                pdf_url = f'https://ftp.ncbi.nlm.nih.gov/pub/pmc/{ftp_path}'

            pdf_path = os.path.join(pdf_dir, f"{paper.clean_pmcid}.pdf")

            logger.debug(f"Downloading {pdf_url} -> {pdf_path}")

            # Create directory if it doesn't exist
            os.makedirs(pdf_dir, exist_ok=True)

            async with session.get(pdf_url) as response:
                if response.status == 200:
                    content = await response.read()

                    # Write PDF file
                    with open(pdf_path, 'wb') as f:
                        f.write(content)

                    # Validate PDF
                    if PDFValidator.is_valid_pdf(pdf_path):
                        paper.pdf_path = pdf_path
                        paper.download_success = True

                        # Convert to text if MarkItDown is available
                        if MARKITDOWN_AVAILABLE:
                            await self._convert_pdf_to_text(pdf_path, paper)

                        logger.info(f"Successfully downloaded {paper.pmcid} PDF via PMC OA")
                        return True
                    else:
                        PDFValidator.cleanup_invalid_pdf(pdf_path)
                        logger.warning(f"Invalid PDF downloaded for {paper.pmcid}")
                        return False
                else:
                    logger.warning(f"Failed to download {pdf_url}: HTTP {response.status}")
                    return False

        except Exception as e:
            logger.error(f"PMC OA download failed for {paper.pmcid}: {e}")
            return False

    async def _convert_pdf_to_text(self, pdf_path: str, paper: Paper) -> None:
        """Convert PDF to text using MarkItDown"""
        try:
            md_converter = MarkItDown(enable_plugins=False)
            result = md_converter.convert(pdf_path)

            txt_path = pdf_path.replace(".pdf", ".txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(result.text_content)

            paper.txt_path = txt_path
            paper.full_text = result.text_content

            logger.debug(f"Converted {pdf_path} -> {txt_path}")

        except Exception as e:
            logger.error(f"MarkItDown conversion failed for {pdf_path}: {e}")


class DirectPMCStrategy(PDFDownloadStrategy):
    """Download PDFs directly from PMC using multiple URL patterns"""

    @property
    def name(self) -> str:
        return "Direct PMC"

    async def can_handle(self, paper: Paper) -> bool:
        """Check if paper has PMC ID"""
        return paper.pmcid is not None

    async def download(self, session: aiohttp.ClientSession, paper: Paper, pdf_dir: str) -> bool:
        """Try multiple PMC URL patterns"""
        if not paper.pmcid:
            return False

        pmcid_clean = paper.clean_pmcid

        # Multiple URL patterns to try
        url_patterns = [
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid_clean}/pdf/",
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid_clean}/pdf/{paper.pmcid}.pdf",
            f"https://europepmc.org/articles/PMC{pmcid_clean}?pdf=render",
        ]

        for url in url_patterns:
            if await self._try_download_url(session, url, paper, pdf_dir):
                return True

        return False

    async def _try_download_url(self, session: aiohttp.ClientSession, url: str, paper: Paper, pdf_dir: str) -> bool:
        """Try downloading from a specific URL"""
        try:
            pdf_path = os.path.join(pdf_dir, f"{paper.clean_pmcid}.pdf")

            logger.debug(f"Trying download: {url} -> {pdf_path}")

            # Add realistic headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/pdf,*/*',
                'Accept-Language': 'en-US,en;q=0.9',
            }

            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '').lower()
                    if 'pdf' in content_type or len(await response.read()) > 1024:

                        # Reset response
                        async with session.get(url, headers=headers) as response2:
                            content = await response2.read()

                            os.makedirs(pdf_dir, exist_ok=True)
                            with open(pdf_path, 'wb') as f:
                                f.write(content)

                            if PDFValidator.is_valid_pdf(pdf_path):
                                paper.pdf_path = pdf_path
                                paper.download_success = True
                                logger.info(f"Downloaded {paper.pmcid} from {url}")
                                return True
                            else:
                                PDFValidator.cleanup_invalid_pdf(pdf_path)

        except Exception as e:
            logger.debug(f"Failed to download from {url}: {e}")

        return False


class UnifiedPDFFetcher:
    """Unified PDF fetcher with multiple download strategies and fallbacks"""

    def __init__(self, pdf_dir: str = "pdfs", markdown_dir: str = "markdown"):
        self.pdf_dir = pdf_dir
        self.markdown_dir = markdown_dir

        # Initialize markdown converter
        self.markdown_converter = MarkdownConverter(markdown_dir)

        # Order strategies by reliability/preference
        self.strategies = [
            PMCOAServiceStrategy(),  # Your ftp_pubmed.py approach - most reliable
            DirectPMCStrategy(),     # Fallback to direct PMC
        ]

    async def download_all(self, papers: List[Paper], batch_size: int = 8) -> None:
        """
        Download PDFs for all papers using optimized multi-strategy approach

        Args:
            papers: List of Paper objects
            batch_size: Number of concurrent downloads (increased for speed)
        """
        # Filter papers that might have downloadable PDFs
        downloadable_papers = [p for p in papers if p.pmcid or p.doi]

        if not downloadable_papers:
            logger.warning("No papers with PMC IDs or DOIs found - skipping PDF download")
            return

        logger.info(f"Attempting to download PDFs for {len(downloadable_papers)} papers")

        # Optimized session settings for faster downloads
        connector = aiohttp.TCPConnector(
            limit=20,              # Increased connection pool
            limit_per_host=8,      # More connections per host
            ttl_dns_cache=300,     # Cache DNS lookups
            use_dns_cache=True,
            keepalive_timeout=30,  # Keep connections alive longer
            enable_cleanup_closed=True
        )
        timeout = aiohttp.ClientTimeout(total=45, connect=10)  # Faster timeouts

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Process in larger batches for speed
            for i in range(0, len(downloadable_papers), batch_size):
                batch = downloadable_papers[i : i + batch_size]

                tasks = [self._download_single_paper(session, paper) for paper in batch]
                await asyncio.gather(*tasks, return_exceptions=True)

                # Reduced delay between batches for speed
                if i + batch_size < len(downloadable_papers):
                    await asyncio.sleep(0.5)  # Minimal delay to stay under rate limits

        # Report results
        successful_downloads = sum(1 for p in downloadable_papers if p.download_success)
        success_rate = (successful_downloads / len(downloadable_papers)) * 100

        logger.info(f"PDF download results: {successful_downloads}/{len(downloadable_papers)} "
                   f"successful ({success_rate:.1f}%)")

        # Check if we have minimum viable success rate
        if success_rate < 30:
            logger.warning(f"Low PDF download success rate ({success_rate:.1f}%) - "
                          "many analyses will rely on abstracts only")

        # Convert downloaded PDFs to Markdown if MarkItDown is available
        if successful_downloads > 0 and self.markdown_converter.is_available():
            await self._convert_pdfs_to_markdown(downloadable_papers)

    async def _download_single_paper(self, session: aiohttp.ClientSession, paper: Paper) -> None:
        """Download PDF for a single paper using available strategies with parallel attempts"""
        # Try strategies that can handle this paper in parallel for speed
        applicable_strategies = []
        for strategy in self.strategies:
            if await strategy.can_handle(paper):
                applicable_strategies.append(strategy)

        if not applicable_strategies:
            paper.error_message = "No applicable download strategies found"
            logger.debug(f"No strategies can handle {paper.pmcid or paper.pmid}")
            return

        # Try all applicable strategies in parallel (first success wins)
        tasks = []
        for strategy in applicable_strategies:
            tasks.append(asyncio.create_task(self._try_strategy(strategy, session, paper)))

        if tasks:
            # Wait for first successful download
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            # Cancel remaining tasks to save resources
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Check if any succeeded
            for task in done:
                if await task:  # If strategy succeeded
                    return

        # All strategies failed
        paper.error_message = "All download strategies failed"
        logger.debug(f"All download strategies failed for {paper.pmcid or paper.pmid}")

    async def _try_strategy(self, strategy: PDFDownloadStrategy, session: aiohttp.ClientSession, paper: Paper) -> bool:
        """Try a single strategy and return success status"""
        try:
            success = await strategy.download(session, paper, self.pdf_dir)
            if success:
                logger.debug(f"Downloaded {paper.pmcid or paper.pmid} via {strategy.name}")
                return True
        except Exception as e:
            logger.error(f"Strategy {strategy.name} failed for {paper.pmcid or paper.pmid}: {e}")

        return False

    async def _convert_pdfs_to_markdown(self, papers: List[Paper]) -> None:
        """Convert all successfully downloaded PDFs to Markdown format"""
        logger.info("📝 Converting downloaded PDFs to Markdown format...")

        # Use asyncio to run the synchronous conversion in a thread pool
        loop = asyncio.get_event_loop()
        markdown_results = await loop.run_in_executor(
            None,
            self.markdown_converter.convert_papers_pdfs,
            papers
        )

        if markdown_results:
            stats = self.markdown_converter.get_conversion_stats()
            logger.info(f"✅ Markdown conversion complete: {stats['total_files']} files, "
                       f"{stats['total_size_mb']} MB total")
        else:
            logger.warning("❌ No PDFs were converted to Markdown")

    def convert_existing_pdfs_to_markdown(self) -> Dict[str, str]:
        """
        Convert all existing PDFs in the PDF directory to Markdown

        Returns:
            Dictionary mapping PDF filename to markdown file path
        """
        logger.info("🔄 Converting existing PDFs to Markdown...")
        results = self.markdown_converter.convert_pdf_directory(self.pdf_dir)

        if results:
            stats = self.markdown_converter.get_conversion_stats()
            logger.info(f"✅ Existing PDF conversion complete: {stats['total_files']} files, "
                       f"{stats['total_size_mb']} MB total")

        return results