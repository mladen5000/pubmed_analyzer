#!/usr/bin/env python3
"""
Simple API Interface for Robust PDF Fetching
Clean, user-friendly interface for the robust PDF fetching library
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from ..models.paper import Paper
from ..core.search import PubMedSearcher
from ..core.robust_pdf_fetcher import RobustPDFFetcher, BatchResult, DownloadResult

logger = logging.getLogger(__name__)


class PubMedPDFFetcher:
    """
    Simple, production-ready PDF fetching library for PubMed papers

    Features:
    - Multi-strategy PDF downloading with fallbacks
    - Rate limiting and respectful downloading
    - Circuit breakers and error handling
    - Batch processing with success rate monitoring
    - Comprehensive validation and quality checks
    """

    def __init__(self,
                 email: str,
                 api_key: Optional[str] = None,
                 pdf_dir: str = "pdfs",
                 min_success_rate: float = 0.3,
                 batch_size: int = 5):
        """
        Initialize the PDF fetcher

        Args:
            email: Email address for NCBI API (required)
            api_key: NCBI API key for higher rate limits (optional)
            pdf_dir: Directory to store downloaded PDFs
            min_success_rate: Minimum success rate before halting batch operations
            batch_size: Number of papers to download per batch
        """
        self.email = email
        self.api_key = api_key

        # Initialize components
        self.searcher = PubMedSearcher(email, api_key)
        self.pdf_fetcher = RobustPDFFetcher(
            pdf_dir=pdf_dir,
            min_success_rate=min_success_rate,
            batch_size=batch_size
        )

        logger.info(f"Initialized PubMedPDFFetcher with email: {email}")
        if api_key:
            logger.info("Using NCBI API key for higher rate limits")

    async def download_from_pmids(self, pmids: List[str]) -> BatchResult:
        """
        Download PDFs for a list of PMIDs

        Args:
            pmids: List of PubMed IDs

        Returns:
            BatchResult with download statistics and individual results
        """
        if not pmids:
            return BatchResult(
                total_papers=0,
                successful_downloads=0,
                failed_downloads=0,
                success_rate=0.0,
                total_time=0.0,
                results=[]
            )

        # Fetch paper metadata
        logger.info(f"Fetching metadata for {len(pmids)} papers...")
        papers = await self.searcher.fetch_papers_metadata(pmids)

        # Download PDFs
        logger.info(f"Starting PDF downloads for {len(papers)} papers...")
        result = await self.pdf_fetcher.download_batch(papers)

        logger.info(f"Download complete: {result.successful_downloads}/{result.total_papers} "
                   f"({result.success_rate:.1%}) successful in {result.total_time:.1f}s")

        return result

    async def download_from_search(self,
                                  query: str,
                                  max_results: int = 50,
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None) -> BatchResult:
        """
        Search PubMed and download PDFs for results

        Args:
            query: PubMed search query
            max_results: Maximum number of papers to download
            start_date: Start date filter (YYYY/MM/DD format)
            end_date: End date filter (YYYY/MM/DD format)

        Returns:
            BatchResult with download statistics and individual results
        """
        logger.info(f"Searching PubMed for: '{query}'")

        # Search for papers
        pmids = await self.searcher.search_papers(
            query=query,
            max_results=max_results,
            start_date=start_date,
            end_date=end_date
        )

        if not pmids:
            logger.warning("No papers found for the query")
            return BatchResult(
                total_papers=0,
                successful_downloads=0,
                failed_downloads=0,
                success_rate=0.0,
                total_time=0.0,
                results=[]
            )

        return await self.download_from_pmids(pmids)

    async def download_single(self, pmid: str) -> DownloadResult:
        """
        Download PDF for a single PMID

        Args:
            pmid: PubMed ID

        Returns:
            DownloadResult for the single paper
        """
        logger.info(f"Downloading PDF for PMID: {pmid}")

        # Fetch metadata
        papers = await self.searcher.fetch_papers_metadata([pmid])
        if not papers:
            return DownloadResult(
                pmid=pmid,
                success=False,
                error_message="Could not fetch paper metadata"
            )

        paper = papers[0]
        result = await self.pdf_fetcher.download_single(paper)

        if result.success:
            logger.info(f"✅ Successfully downloaded {pmid} using {result.strategy_used}")
        else:
            logger.warning(f"❌ Failed to download {pmid}: {result.error_message}")

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about PDF fetching performance"""
        return self.pdf_fetcher.get_statistics()

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of all download strategies"""
        return await self.pdf_fetcher.health_check()


# Synchronous wrapper for easier usage
class PubMedPDFFetcherSync:
    """Synchronous wrapper for PubMedPDFFetcher"""

    def __init__(self, *args, **kwargs):
        self._fetcher = PubMedPDFFetcher(*args, **kwargs)

    def download_from_pmids(self, pmids: List[str]) -> BatchResult:
        """Synchronous version of download_from_pmids"""
        return asyncio.run(self._fetcher.download_from_pmids(pmids))

    def download_from_search(self, query: str, **kwargs) -> BatchResult:
        """Synchronous version of download_from_search"""
        return asyncio.run(self._fetcher.download_from_search(query, **kwargs))

    def download_single(self, pmid: str) -> DownloadResult:
        """Synchronous version of download_single"""
        return asyncio.run(self._fetcher.download_single(pmid))

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return self._fetcher.get_statistics()

    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return asyncio.run(self._fetcher.health_check())


# Convenience functions for quick usage
async def download_pdfs_by_pmids(pmids: List[str],
                                email: str,
                                api_key: Optional[str] = None,
                                pdf_dir: str = "pdfs") -> BatchResult:
    """
    Quick function to download PDFs by PMIDs

    Args:
        pmids: List of PubMed IDs
        email: Email for NCBI API
        api_key: Optional NCBI API key
        pdf_dir: Directory to save PDFs

    Returns:
        BatchResult with download statistics
    """
    fetcher = PubMedPDFFetcher(email=email, api_key=api_key, pdf_dir=pdf_dir)
    return await fetcher.download_from_pmids(pmids)


async def download_pdfs_by_search(query: str,
                                 email: str,
                                 max_results: int = 50,
                                 api_key: Optional[str] = None,
                                 pdf_dir: str = "pdfs") -> BatchResult:
    """
    Quick function to search and download PDFs

    Args:
        query: PubMed search query
        email: Email for NCBI API
        max_results: Maximum papers to download
        api_key: Optional NCBI API key
        pdf_dir: Directory to save PDFs

    Returns:
        BatchResult with download statistics
    """
    fetcher = PubMedPDFFetcher(email=email, api_key=api_key, pdf_dir=pdf_dir)
    return await fetcher.download_from_search(query, max_results=max_results)


def download_pdfs_sync(pmids: List[str],
                      email: str,
                      api_key: Optional[str] = None,
                      pdf_dir: str = "pdfs") -> BatchResult:
    """
    Synchronous convenience function for PDF downloads

    Args:
        pmids: List of PubMed IDs
        email: Email for NCBI API
        api_key: Optional NCBI API key
        pdf_dir: Directory to save PDFs

    Returns:
        BatchResult with download statistics
    """
    return asyncio.run(download_pdfs_by_pmids(pmids, email, api_key, pdf_dir))