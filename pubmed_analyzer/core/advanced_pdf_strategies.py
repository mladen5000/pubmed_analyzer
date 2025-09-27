#!/usr/bin/env python3
"""
Advanced PDF Fetching Strategies for Non-Open Access Articles
Implements high-success rate strategies for paywalled and subscription-based content
"""

import asyncio
import aiohttp
import logging
import tempfile
import json
import time
from typing import Optional, Dict, Any, List
from pathlib import Path

from .robust_pdf_fetcher import PDFDownloadStrategy, DownloadResult
from ..models.paper import Paper
from ..utils.validators import PDFValidator

logger = logging.getLogger(__name__)


class SemanticScholarStrategy(PDFDownloadStrategy):
    """Semantic Scholar API - High success rate for academic papers with direct PDF access"""

    def __init__(self):
        super().__init__()
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper"
        self.rate_limit_delay = 1.0  # 1 second between requests (respectful rate limiting)

    @property
    def name(self) -> str:
        return "Semantic Scholar API"

    @property
    def priority(self) -> int:
        return 8  # After existing strategies but before browser automation

    async def can_handle(self, paper: Paper) -> bool:
        """Can handle papers with DOI, PMID, or title"""
        return bool(paper.doi or paper.pmid or paper.title)

    async def get_pdf_url(self, paper: Paper) -> Optional[str]:
        """Get PDF URL from Semantic Scholar API"""
        try:
            # Try DOI first (most reliable)
            if paper.doi:
                search_id = f"DOI:{paper.doi}"
            elif paper.pmid:
                # Remove PMID: prefix if present
                pmid_clean = paper.pmid.replace("PMID:", "").strip()
                search_id = f"PMID:{pmid_clean}"
            else:
                # Fallback to title search (less reliable)
                return None

            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)

            return f"SEMANTIC_SCHOLAR:{search_id}"

        except Exception as e:
            logger.debug(f"Semantic Scholar URL generation failed: {e}")
            return None

    async def download_pdf(self, session: aiohttp.ClientSession, paper: Paper, pdf_dir: str, rate_limiter) -> DownloadResult:
        """Download PDF using Semantic Scholar API"""
        try:
            pdf_url = await self.get_pdf_url(paper)
            if not pdf_url:
                return DownloadResult(
                    pmid=paper.pmid,
                    success=False,
                    strategy_used=self.name,
                    error_message="No valid search identifier"
                )

            search_id = pdf_url.replace("SEMANTIC_SCHOLAR:", "")

            # Query Semantic Scholar API
            api_url = f"{self.base_url}/{search_id}"
            params = {
                "fields": "title,openAccessPdf,url,isOpenAccess,publicationTypes"
            }

            headers = {
                "User-Agent": "Academic Research Tool (educational use)",
                "Accept": "application/json"
            }

            async with session.get(api_url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()

                    # Check if paper has open access PDF
                    if data.get("openAccessPdf") and data["openAccessPdf"].get("url"):
                        pdf_download_url = data["openAccessPdf"]["url"]

                        # Download the PDF
                        async with session.get(pdf_download_url, headers=headers) as pdf_response:
                            if pdf_response.status == 200:
                                content = await pdf_response.read()

                                # Save PDF
                                pdf_path = Path(pdf_dir) / f"{paper.pmid}.pdf"
                                pdf_path.parent.mkdir(parents=True, exist_ok=True)

                                with open(pdf_path, 'wb') as f:
                                    f.write(content)

                                # Validate PDF
                                if PDFValidator.is_valid_pdf(str(pdf_path)):
                                    self.circuit_breaker.record_success()
                                    self.success_count += 1
                                    return DownloadResult(
                                        pmid=paper.pmid,
                                        success=True,
                                        file_path=str(pdf_path),
                                        file_size=len(content),
                                        strategy_used=self.name,
                                        metadata={
                                            'semantic_scholar_id': data.get('paperId'),
                                            'is_open_access': data.get('isOpenAccess'),
                                            'publication_types': data.get('publicationTypes', [])
                                        }
                                    )
                                else:
                                    PDFValidator.cleanup_invalid_pdf(str(pdf_path))
                                    return DownloadResult(
                                        pmid=paper.pmid,
                                        success=False,
                                        strategy_used=self.name,
                                        error_message="Invalid PDF content"
                                    )
                            else:
                                return DownloadResult(
                                    pmid=paper.pmid,
                                    success=False,
                                    strategy_used=self.name,
                                    error_message=f"PDF download failed: HTTP {pdf_response.status}"
                                )
                    else:
                        return DownloadResult(
                            pmid=paper.pmid,
                            success=False,
                            strategy_used=self.name,
                            error_message="No open access PDF available"
                        )
                elif response.status == 429:
                    # Rate limited
                    await asyncio.sleep(5)  # Wait longer if rate limited
                    return DownloadResult(
                        pmid=paper.pmid,
                        success=False,
                        strategy_used=self.name,
                        error_message="Rate limited by Semantic Scholar API"
                    )
                else:
                    return DownloadResult(
                        pmid=paper.pmid,
                        success=False,
                        strategy_used=self.name,
                        error_message=f"API error: HTTP {response.status}"
                    )

        except Exception as e:
            self.circuit_breaker.record_failure()
            self.failure_count += 1
            return DownloadResult(
                pmid=paper.pmid,
                success=False,
                strategy_used=self.name,
                error_message=f"Semantic Scholar error: {str(e)}"
            )


class COREStrategy(PDFDownloadStrategy):
    """CORE.ac.uk API - 28M+ papers from institutional repositories"""

    def __init__(self):
        super().__init__()
        self.base_url = "https://api.core.ac.uk/v3"
        self.rate_limit_delay = 0.5  # 500ms between requests

    @property
    def name(self) -> str:
        return "CORE.ac.uk API"

    @property
    def priority(self) -> int:
        return 9  # After Semantic Scholar

    async def can_handle(self, paper: Paper) -> bool:
        """Can handle papers with DOI or title"""
        return bool(paper.doi or paper.title)

    async def get_pdf_url(self, paper: Paper) -> Optional[str]:
        """Get search query for CORE API"""
        if paper.doi:
            return f"CORE_DOI:{paper.doi}"
        elif paper.title:
            return f"CORE_TITLE:{paper.title}"
        return None

    async def download_pdf(self, session: aiohttp.ClientSession, paper: Paper, pdf_dir: str, rate_limiter) -> DownloadResult:
        """Download PDF using CORE API"""
        try:
            pdf_url = await self.get_pdf_url(paper)
            if not pdf_url:
                return DownloadResult(
                    pmid=paper.pmid,
                    success=False,
                    strategy_used=self.name,
                    error_message="No search criteria available"
                )

            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)

            if pdf_url.startswith("CORE_DOI:"):
                doi = pdf_url.replace("CORE_DOI:", "")
                search_query = f'doi:"{doi}"'
            else:
                title = pdf_url.replace("CORE_TITLE:", "")
                search_query = f'title:"{title}"'

            # Search CORE API
            search_url = f"{self.base_url}/search/works"
            params = {
                "q": search_query,
                "limit": 5,
                "scroll": "false",
                "stats": "false"
            }

            headers = {
                "User-Agent": "Academic Research Tool (educational use)",
                "Accept": "application/json"
            }

            async with session.get(search_url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get("results", [])

                    for result in results:
                        # Look for download URL
                        download_url = result.get("downloadUrl")
                        if download_url and download_url.endswith('.pdf'):
                            # Try to download the PDF
                            async with session.get(download_url, headers=headers) as pdf_response:
                                if pdf_response.status == 200:
                                    content = await pdf_response.read()

                                    # Save PDF
                                    pdf_path = Path(pdf_dir) / f"{paper.pmid}.pdf"
                                    pdf_path.parent.mkdir(parents=True, exist_ok=True)

                                    with open(pdf_path, 'wb') as f:
                                        f.write(content)

                                    # Validate PDF
                                    if PDFValidator.is_valid_pdf(str(pdf_path)):
                                        self.circuit_breaker.record_success()
                                        self.success_count += 1
                                        return DownloadResult(
                                            pmid=paper.pmid,
                                            success=True,
                                            file_path=str(pdf_path),
                                            file_size=len(content),
                                            strategy_used=self.name,
                                            metadata={
                                                'core_id': result.get('id'),
                                                'repository': result.get('repositories', [{}])[0].get('name', 'Unknown'),
                                                'year_published': result.get('yearPublished')
                                            }
                                        )
                                    else:
                                        PDFValidator.cleanup_invalid_pdf(str(pdf_path))

                    return DownloadResult(
                        pmid=paper.pmid,
                        success=False,
                        strategy_used=self.name,
                        error_message="No downloadable PDFs found in CORE"
                    )
                else:
                    return DownloadResult(
                        pmid=paper.pmid,
                        success=False,
                        strategy_used=self.name,
                        error_message=f"CORE API error: HTTP {response.status}"
                    )

        except Exception as e:
            self.circuit_breaker.record_failure()
            self.failure_count += 1
            return DownloadResult(
                pmid=paper.pmid,
                success=False,
                strategy_used=self.name,
                error_message=f"CORE error: {str(e)}"
            )


class UnpaywallStrategy(PDFDownloadStrategy):
    """Unpaywall API - Legal open access discovery service"""

    def __init__(self):
        super().__init__()
        self.base_url = "https://api.unpaywall.org/v2"
        self.rate_limit_delay = 1.0  # 1 second between requests

    @property
    def name(self) -> str:
        return "Unpaywall API"

    @property
    def priority(self) -> int:
        return 10  # After institutional repositories

    async def can_handle(self, paper: Paper) -> bool:
        """Only works with DOIs"""
        return bool(paper.doi)

    async def get_pdf_url(self, paper: Paper) -> Optional[str]:
        """Get DOI for Unpaywall lookup"""
        return f"UNPAYWALL:{paper.doi}" if paper.doi else None

    async def download_pdf(self, session: aiohttp.ClientSession, paper: Paper, pdf_dir: str, rate_limiter) -> DownloadResult:
        """Download PDF using Unpaywall API"""
        try:
            if not paper.doi:
                return DownloadResult(
                    pmid=paper.pmid,
                    success=False,
                    strategy_used=self.name,
                    error_message="No DOI available"
                )

            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)

            # Query Unpaywall API
            api_url = f"{self.base_url}/{paper.doi}"
            params = {
                "email": "research@academic.edu"  # Required by Unpaywall
            }

            headers = {
                "User-Agent": "Academic Research Tool (educational use)",
                "Accept": "application/json"
            }

            async with session.get(api_url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()

                    # Check if open access version is available
                    if data.get("is_oa") and data.get("best_oa_location"):
                        oa_location = data["best_oa_location"]
                        pdf_url = oa_location.get("url_for_pdf")

                        if pdf_url:
                            # Download the PDF
                            async with session.get(pdf_url, headers=headers) as pdf_response:
                                if pdf_response.status == 200:
                                    content = await pdf_response.read()

                                    # Save PDF
                                    pdf_path = Path(pdf_dir) / f"{paper.pmid}.pdf"
                                    pdf_path.parent.mkdir(parents=True, exist_ok=True)

                                    with open(pdf_path, 'wb') as f:
                                        f.write(content)

                                    # Validate PDF
                                    if PDFValidator.is_valid_pdf(str(pdf_path)):
                                        self.circuit_breaker.record_success()
                                        self.success_count += 1
                                        return DownloadResult(
                                            pmid=paper.pmid,
                                            success=True,
                                            file_path=str(pdf_path),
                                            file_size=len(content),
                                            strategy_used=self.name,
                                            metadata={
                                                'oa_date': data.get('oa_date'),
                                                'host_type': oa_location.get('host_type'),
                                                'license': oa_location.get('license')
                                            }
                                        )
                                    else:
                                        PDFValidator.cleanup_invalid_pdf(str(pdf_path))

                    return DownloadResult(
                        pmid=paper.pmid,
                        success=False,
                        strategy_used=self.name,
                        error_message="No open access PDF available via Unpaywall"
                    )
                else:
                    return DownloadResult(
                        pmid=paper.pmid,
                        success=False,
                        strategy_used=self.name,
                        error_message=f"Unpaywall API error: HTTP {response.status}"
                    )

        except Exception as e:
            self.circuit_breaker.record_failure()
            self.failure_count += 1
            return DownloadResult(
                pmid=paper.pmid,
                success=False,
                strategy_used=self.name,
                error_message=f"Unpaywall error: {str(e)}"
            )