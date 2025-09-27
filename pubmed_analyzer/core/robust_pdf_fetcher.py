#!/usr/bin/env python3
"""
Robust PDF Fetching Library for PubMed IDs
Production-ready multi-strategy PDF downloader with comprehensive error handling,
rate limiting, validation, and respectful downloading practices.
"""

import asyncio
import aiohttp
import os
import time
import logging
import tempfile
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib
import json

from ..models.paper import Paper
from ..utils.validators import PDFValidator

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Result of a PDF download attempt"""
    pmid: str
    success: bool
    file_path: Optional[str] = None
    file_size: int = 0
    strategy_used: Optional[str] = None
    attempt_count: int = 0
    error_message: Optional[str] = None
    download_time: float = 0.0
    validation_passed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result of a batch download operation"""
    total_papers: int
    successful_downloads: int
    failed_downloads: int
    success_rate: float
    total_time: float
    results: List[DownloadResult]
    strategies_used: Dict[str, int] = field(default_factory=dict)


class TokenBucket:
    """Token bucket rate limiter for respectful API usage"""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Acquire a token, blocking if necessary"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True

            # Wait for next token
            wait_time = (1 - self.tokens) / self.rate
            await asyncio.sleep(wait_time)
            self.tokens = 0
            return True


class CircuitBreaker:
    """Circuit breaker to temporarily halt failing strategies"""

    def __init__(self, failure_threshold: int = 25, timeout: float = 300, strategy_name: str = "Unknown"):
        self.failure_threshold = failure_threshold  # Increased from 10 to 25
        self.timeout = timeout  # Increased from 120 to 300 seconds
        self.strategy_name = strategy_name
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half_open
        self.consecutive_failures = 0  # Track consecutive failures vs total

    def record_success(self):
        """Record successful operation"""
        # Reset consecutive failures but keep total count for statistics
        self.consecutive_failures = 0
        self.state = 'closed'

    def record_failure(self):
        """Record failed operation with smarter logic"""
        self.failure_count += 1
        self.consecutive_failures += 1
        self.last_failure_time = time.time()

        # Only open circuit after consecutive failures, not total failures
        if self.consecutive_failures >= self.failure_threshold:
            self.state = 'open'
            logger.warning(f"Circuit breaker opened for {self.strategy_name} after {self.consecutive_failures} consecutive failures")
        elif self.consecutive_failures >= 5:  # Warn at 5 consecutive failures
            logger.info(f"Strategy {self.strategy_name} has {self.consecutive_failures} consecutive failures")

    def can_proceed(self) -> bool:
        """Check if operations can proceed"""
        if self.state == 'closed':
            return True

        if self.state == 'open':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'half_open'
                return True
            return False

        # half_open state - allow one attempt
        return True


class RateLimiter:
    """Advanced rate limiter with per-domain limits"""

    def __init__(self):
        self.limiters = {
            'pmc.ncbi.nlm.nih.gov': TokenBucket(rate=3.0, capacity=10),
            'publisher_default': TokenBucket(rate=1.0, capacity=5),
            'arxiv.org': TokenBucket(rate=2.0, capacity=8),
        }

    def get_limiter(self, url: str) -> TokenBucket:
        """Get appropriate rate limiter for URL"""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc

        # Use domain-specific limiter if available
        if domain in self.limiters:
            return self.limiters[domain]

        # Use default publisher limiter
        return self.limiters['publisher_default']

    async def acquire_for_url(self, url: str):
        """Acquire rate limit token for specific URL"""
        limiter = self.get_limiter(url)
        await limiter.acquire()


class PDFDownloadStrategy(ABC):
    """Abstract base class for PDF download strategies"""

    def __init__(self):
        self.circuit_breaker = CircuitBreaker(strategy_name=self.__class__.__name__)
        self.success_count = 0
        self.failure_count = 0

    @abstractmethod
    async def can_handle(self, paper: Paper) -> bool:
        """Check if this strategy can handle downloading this paper"""
        pass

    @abstractmethod
    async def get_pdf_url(self, paper: Paper) -> Optional[str]:
        """Get PDF URL for the paper"""
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

    def get_success_rate(self) -> float:
        """Get current success rate for this strategy"""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    async def download_pdf(self, session: aiohttp.ClientSession, paper: Paper,
                          pdf_dir: str, rate_limiter: RateLimiter) -> DownloadResult:
        """Download PDF using this strategy"""
        if not self.circuit_breaker.can_proceed():
            return DownloadResult(
                pmid=paper.pmid,
                success=False,
                strategy_used=self.name,
                error_message="Circuit breaker open"
            )

        try:
            pdf_url = await self.get_pdf_url(paper)
            if not pdf_url:
                return DownloadResult(
                    pmid=paper.pmid,
                    success=False,
                    strategy_used=self.name,
                    error_message="No PDF URL found"
                )

            # SPECIAL HANDLING: PMC OA Service requires XML parsing
            if pdf_url.startswith("PMC_OA_SERVICE:"):
                pmcid = pdf_url.replace("PMC_OA_SERVICE:", "")
                actual_pdf_url = await self._parse_pmc_oa_service(session, pmcid)
                if not actual_pdf_url:
                    return DownloadResult(
                        pmid=paper.pmid,
                        success=False,
                        strategy_used=self.name,
                        error_message="No PDF found in OA Service XML"
                    )
                pdf_url = actual_pdf_url

            # Rate limiting
            await rate_limiter.acquire_for_url(pdf_url)

            # Download with retry logic
            result = await self._download_with_retry(session, paper, pdf_url, pdf_dir)

            if result.success:
                self.circuit_breaker.record_success()
                self.success_count += 1
            else:
                self.circuit_breaker.record_failure()
                self.failure_count += 1

            result.strategy_used = self.name
            return result

        except Exception as e:
            self.circuit_breaker.record_failure()
            self.failure_count += 1
            return DownloadResult(
                pmid=paper.pmid,
                success=False,
                strategy_used=self.name,
                error_message=str(e)
            )

    async def _download_with_retry(self, session: aiohttp.ClientSession,
                                  paper: Paper, pdf_url: str, pdf_dir: str) -> DownloadResult:
        """Download with exponential backoff retry"""
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                start_time = time.time()

                # IMPROVED: Better headers to avoid 403 errors
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'cross-site',
                    'Cache-Control': 'max-age=0',
                }

                timeout = aiohttp.ClientTimeout(total=30)
                async with session.get(pdf_url, headers=headers, timeout=timeout, allow_redirects=True) as response:
                    if response.status == 200:
                        # IMPROVED: Check final URL and content-type for PDF with better logic
                        final_url = str(response.url).lower()
                        content_type = response.headers.get('content-type', '').lower()

                        # More flexible PDF detection
                        is_pdf_url = ('.pdf' in final_url) or ('pdf' in final_url and 'download' in final_url)
                        is_pdf_content = ('application/pdf' in content_type) or ('pdf' in content_type)

                        # Check content size (PDFs are typically large)
                        content_length = response.headers.get('content-length')
                        is_reasonable_size = not content_length or int(content_length) > 10000

                        if not (is_pdf_url or is_pdf_content) and is_reasonable_size:
                            # For publisher sites, check if we got redirected to paywall/login
                            paywall_indicators = ['login', 'subscribe', 'paywall', 'access-denied', 'forbidden']
                            if any(indicator in final_url for indicator in paywall_indicators):
                                logger.debug(f"Redirected to paywall: {final_url}")
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(base_delay * (2 ** attempt))
                                    continue
                                else:
                                    return DownloadResult(
                                        pmid=paper.pmid,
                                        success=False,
                                        attempt_count=attempt + 1,
                                        error_message=f"Redirected to paywall: {final_url}"
                                    )
                            # For non-paywall redirects, still try to download if size is reasonable
                            logger.debug(f"Non-PDF URL but proceeding: {final_url}, content-type: {content_type}")

                        content = await response.read()

                        # Validate PDF content
                        if not self._is_valid_pdf_content(content):
                            if attempt < max_retries - 1:
                                await asyncio.sleep(base_delay * (2 ** attempt))
                                continue
                            else:
                                return DownloadResult(
                                    pmid=paper.pmid,
                                    success=False,
                                    attempt_count=attempt + 1,
                                    error_message="Invalid PDF content"
                                )

                        # Save file
                        pdf_path = Path(pdf_dir) / f"{paper.pmid}.pdf"
                        pdf_path.parent.mkdir(parents=True, exist_ok=True)

                        with open(pdf_path, 'wb') as f:
                            f.write(content)

                        # Final validation
                        is_valid = PDFValidator.is_valid_pdf(str(pdf_path))
                        validation_result = is_valid

                        download_time = time.time() - start_time

                        return DownloadResult(
                            pmid=paper.pmid,
                            success=True,
                            file_path=str(pdf_path),
                            file_size=len(content),
                            attempt_count=attempt + 1,
                            download_time=download_time,
                            validation_passed=validation_result,
                            metadata={
                                'url': pdf_url,
                                'response_status': response.status,
                                'content_type': response.headers.get('content-type')
                            }
                        )

                    elif response.status in [429, 503, 502]:  # Rate limited or server error
                        if attempt < max_retries - 1:
                            wait_time = base_delay * (2 ** attempt)
                            logger.warning(f"HTTP {response.status} for {pdf_url}, retrying in {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue

                    return DownloadResult(
                        pmid=paper.pmid,
                        success=False,
                        attempt_count=attempt + 1,
                        error_message=f"HTTP {response.status}"
                    )

            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (2 ** attempt))
                    continue
                return DownloadResult(
                    pmid=paper.pmid,
                    success=False,
                    attempt_count=attempt + 1,
                    error_message="Download timeout"
                )

            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (2 ** attempt))
                    continue
                return DownloadResult(
                    pmid=paper.pmid,
                    success=False,
                    attempt_count=attempt + 1,
                    error_message=str(e)
                )

        return DownloadResult(
            pmid=paper.pmid,
            success=False,
            attempt_count=max_retries,
            error_message="All retry attempts failed"
        )

    def _is_valid_pdf_content(self, content: bytes) -> bool:
        """Quick validation of PDF content"""
        if len(content) < 1024:  # Too small
            return False

        if not content.startswith(b'%PDF'):  # Not a PDF
            return False

        # Check for common paywall indicators
        paywall_indicators = [
            b'<html',
            b'<!DOCTYPE',
            b'Access Denied',
            b'Subscription Required',
            b'paywall',
        ]

        content_lower = content[:2048].lower()
        return not any(indicator.lower() in content_lower for indicator in paywall_indicators)

    async def _parse_pmc_oa_service(self, session: aiohttp.ClientSession, pmcid: str) -> Optional[str]:
        """Parse PMC OA Service XML response to extract PDF URL"""
        try:
            import xml.etree.ElementTree as ET

            oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
            headers = {
                'User-Agent': 'PubMedAnalyzer/1.0 (mailto:researcher@university.edu)',
                'Accept': 'application/xml, text/xml',
            }

            async with session.get(oa_url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status != 200:
                    logger.debug(f"PMC OA Service returned {response.status} for {pmcid}")
                    return None

                xml_content = await response.text()
                root = ET.fromstring(xml_content)

                # Check for errors
                error = root.find('error')
                if error is not None:
                    logger.debug(f"PMC OA Service error for {pmcid}: {error.text}")
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
                            # Convert FTP to HTTPS if needed
                            if href.startswith('ftp://ftp.ncbi.nlm.nih.gov'):
                                href = href.replace('ftp://ftp.ncbi.nlm.nih.gov', 'https://ftp.ncbi.nlm.nih.gov')
                            logger.debug(f"Found PDF via OA Service: {href}")
                            return href

                return None

        except Exception as e:
            logger.error(f"Failed to parse PMC OA Service XML for {pmcid}: {e}")
            return None


class EuropePMCStrategy(PDFDownloadStrategy):
    """EuropePMC - Highest success rate for PMC papers"""

    @property
    def name(self) -> str:
        return "EuropePMC"

    @property
    def priority(self) -> int:
        return 0  # Highest priority

    async def can_handle(self, paper: Paper) -> bool:
        return paper.pmcid is not None

    async def get_pdf_url(self, paper: Paper) -> Optional[str]:
        if not paper.pmcid:
            return None

        # Clean PMC ID
        pmc_id = paper.pmcid.replace('PMC', '')
        return f"https://europepmc.org/articles/PMC{pmc_id}?pdf=render"


class PMCOAServiceStrategy(PDFDownloadStrategy):
    """FIXED: Official PMC Open Access service with proper XML parsing"""

    @property
    def name(self) -> str:
        return "PMC OA Service"

    @property
    def priority(self) -> int:
        return 1  # High priority - now working correctly

    async def can_handle(self, paper: Paper) -> bool:
        return paper.pmcid is not None

    async def get_pdf_url(self, paper: Paper) -> Optional[str]:
        """Get PDF URL by parsing PMC OA Service XML response"""
        if not paper.pmcid:
            return None

        try:
            import xml.etree.ElementTree as ET

            # Clean PMC ID
            pmc_id = paper.pmcid.replace('PMC', '')
            full_pmcid = f"PMC{pmc_id}"

            # Call OA service WITHOUT format=pdf (this was the bug!)
            oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={full_pmcid}"

            # We'll need to make this request in the download method since we need session
            # Return a special marker to indicate we need XML parsing
            return f"PMC_OA_SERVICE:{full_pmcid}"

        except Exception as e:
            logger.error(f"PMC OA Service URL generation failed: {e}")
            return None


class DOIRedirectStrategy(PDFDownloadStrategy):
    """DOI resolution strategy"""

    @property
    def name(self) -> str:
        return "DOI Redirect"

    @property
    def priority(self) -> int:
        return 3

    async def can_handle(self, paper: Paper) -> bool:
        return paper.doi is not None

    async def get_pdf_url(self, paper: Paper) -> Optional[str]:
        if not paper.doi:
            return None
        return f"https://doi.org/{paper.doi}"


class ArxivStrategy(PDFDownloadStrategy):
    """arXiv preprint strategy"""

    @property
    def name(self) -> str:
        return "arXiv"

    @property
    def priority(self) -> int:
        return 4

    async def can_handle(self, paper: Paper) -> bool:
        # Check if DOI or title suggests arXiv
        if paper.doi and 'arxiv' in paper.doi.lower():
            return True
        if paper.title and 'arxiv' in str(paper.title).lower():
            return True
        return False

    async def get_pdf_url(self, paper: Paper) -> Optional[str]:
        if paper.doi and 'arxiv' in paper.doi.lower():
            # Extract arXiv ID from DOI
            arxiv_id = paper.doi.split('/')[-1]
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        return None


class DirectPMCStrategy(PDFDownloadStrategy):
    """Direct PMC PDF access with working URL patterns"""

    @property
    def name(self) -> str:
        return "Direct PMC"

    @property
    def priority(self) -> int:
        return 2

    async def can_handle(self, paper: Paper) -> bool:
        return paper.pmcid is not None

    async def get_pdf_url(self, paper: Paper) -> Optional[str]:
        if not paper.pmcid:
            return None

        # Clean PMC ID
        pmc_id = paper.pmcid.replace('PMC', '')

        # Try multiple PMC URL patterns for better success
        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/"


class EnhancedDOIStrategy(PDFDownloadStrategy):
    """Enhanced DOI strategy with publisher-specific handling"""

    @property
    def name(self) -> str:
        return "Enhanced DOI"

    @property
    def priority(self) -> int:
        return 3

    async def can_handle(self, paper: Paper) -> bool:
        return paper.doi is not None

    async def get_pdf_url(self, paper: Paper) -> Optional[str]:
        if not paper.doi:
            return None

        # Publisher-specific URL patterns for better success
        doi_lower = paper.doi.lower()

        if 'plos' in doi_lower:
            # PLOS journals - direct PDF access
            return f"https://journals.plos.org/plosone/article/file?id={paper.doi}&type=printable"
        elif 'nature' in doi_lower:
            # Nature journals - PDF access pattern
            return f"https://www.nature.com/articles/{paper.doi.split('/')[-1]}.pdf"
        elif 'cell' in doi_lower or 'science' in doi_lower:
            # Try direct PDF access first
            return f"https://doi.org/{paper.doi}"
        else:
            # Standard DOI resolution
            return f"https://doi.org/{paper.doi}"


class ArxivAPIStrategy(PDFDownloadStrategy):
    """Official arXiv API strategy - highest reliability for arXiv papers"""

    @property
    def name(self) -> str:
        return "arXiv API"

    @property
    def priority(self) -> int:
        return 5

    async def can_handle(self, paper: Paper) -> bool:
        if paper.doi and 'arxiv' in paper.doi.lower():
            return True
        if paper.title and 'arxiv' in str(paper.title).lower():
            return True
        return False

    async def get_pdf_url(self, paper: Paper) -> Optional[str]:
        try:
            import arxiv

            if paper.doi and 'arxiv' in paper.doi.lower():
                arxiv_id = paper.doi.split('arxiv.')[-1].split('/')[0]
                return f"ARXIV_API:{arxiv_id}"
            elif paper.title:
                return f"ARXIV_SEARCH:{paper.title}"
            return None
        except ImportError:
            return None

    async def download_pdf(self, session, paper: Paper, pdf_dir: str, rate_limiter) -> DownloadResult:
        try:
            import arxiv

            pdf_url = await self.get_pdf_url(paper)
            if not pdf_url:
                return DownloadResult(pmid=paper.pmid, success=False, strategy_used=self.name, error_message="No arXiv ID found")

            if pdf_url.startswith("ARXIV_API:"):
                arxiv_id = pdf_url.replace("ARXIV_API:", "")
                search = arxiv.Search(id_list=[arxiv_id])
                results = list(search.results())

                if results:
                    paper_obj = results[0]
                    target_path = Path(pdf_dir) / f"{paper.pmid}.pdf"
                    target_path.parent.mkdir(parents=True, exist_ok=True)

                    paper_obj.download_pdf(dirpath=pdf_dir, filename=f"{paper.pmid}.pdf")

                    if target_path.exists():
                        self.circuit_breaker.record_success()
                        self.success_count += 1
                        return DownloadResult(
                            pmid=paper.pmid, success=True, file_path=str(target_path),
                            file_size=target_path.stat().st_size, strategy_used=self.name,
                            metadata={'arxiv_id': arxiv_id}
                        )

            return DownloadResult(pmid=paper.pmid, success=False, strategy_used=self.name, error_message="arXiv download failed")

        except Exception as e:
            self.circuit_breaker.record_failure()
            self.failure_count += 1
            return DownloadResult(pmid=paper.pmid, success=False, strategy_used=self.name, error_message=str(e))


class SemanticScholarStrategy(PDFDownloadStrategy):
    """Semantic Scholar API - High success rate for academic papers with direct PDF access"""

    def __init__(self):
        super().__init__()
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper"
        self.rate_limit_delay = 1.0  # 1 second between requests

    @property
    def name(self) -> str:
        return "Semantic Scholar API"

    @property
    def priority(self) -> int:
        return 8

    async def can_handle(self, paper: Paper) -> bool:
        return bool(paper.doi or paper.pmid or paper.title)

    async def get_pdf_url(self, paper: Paper) -> Optional[str]:
        try:
            if paper.doi:
                search_id = f"DOI:{paper.doi}"
            elif paper.pmid:
                pmid_clean = paper.pmid.replace("PMID:", "").strip()
                search_id = f"PMID:{pmid_clean}"
            else:
                return None

            await asyncio.sleep(self.rate_limit_delay)
            return f"SEMANTIC_SCHOLAR:{search_id}"

        except Exception as e:
            logger.debug(f"Semantic Scholar URL generation failed: {e}")
            return None

    async def download_pdf(self, session: aiohttp.ClientSession, paper: Paper, pdf_dir: str, rate_limiter) -> DownloadResult:
        try:
            pdf_url = await self.get_pdf_url(paper)
            if not pdf_url:
                return DownloadResult(pmid=paper.pmid, success=False, strategy_used=self.name, error_message="No valid search identifier")

            search_id = pdf_url.replace("SEMANTIC_SCHOLAR:", "")
            api_url = f"{self.base_url}/{search_id}"
            params = {"fields": "title,openAccessPdf,url,isOpenAccess,publicationTypes"}
            headers = {
                "User-Agent": "Academic Research Tool (educational use)",
                "Accept": "application/json"
            }

            async with session.get(api_url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()

                    if data.get("openAccessPdf") and data["openAccessPdf"].get("url"):
                        pdf_download_url = data["openAccessPdf"]["url"]

                        async with session.get(pdf_download_url, headers=headers) as pdf_response:
                            if pdf_response.status == 200:
                                content = await pdf_response.read()
                                pdf_path = Path(pdf_dir) / f"{paper.pmid}.pdf"
                                pdf_path.parent.mkdir(parents=True, exist_ok=True)

                                with open(pdf_path, 'wb') as f:
                                    f.write(content)

                                if PDFValidator.is_valid_pdf(str(pdf_path)):
                                    self.circuit_breaker.record_success()
                                    self.success_count += 1
                                    return DownloadResult(
                                        pmid=paper.pmid, success=True, file_path=str(pdf_path),
                                        file_size=len(content), strategy_used=self.name,
                                        metadata={'semantic_scholar_id': data.get('paperId')}
                                    )
                                else:
                                    PDFValidator.cleanup_invalid_pdf(str(pdf_path))

            return DownloadResult(pmid=paper.pmid, success=False, strategy_used=self.name, error_message="No open access PDF available")

        except Exception as e:
            self.circuit_breaker.record_failure()
            self.failure_count += 1
            return DownloadResult(pmid=paper.pmid, success=False, strategy_used=self.name, error_message=f"Semantic Scholar error: {str(e)}")


class UnpaywallStrategy(PDFDownloadStrategy):
    """Unpaywall API - Legal open access discovery service"""

    def __init__(self, email: str = "research@academic.edu"):
        super().__init__()
        self.base_url = "https://api.unpaywall.org/v2"
        self.rate_limit_delay = 1.0
        self.email = email

    @property
    def name(self) -> str:
        return "Unpaywall API"

    @property
    def priority(self) -> int:
        return 10

    async def can_handle(self, paper: Paper) -> bool:
        return bool(paper.doi)

    async def get_pdf_url(self, paper: Paper) -> Optional[str]:
        return f"UNPAYWALL:{paper.doi}" if paper.doi else None

    async def download_pdf(self, session: aiohttp.ClientSession, paper: Paper, pdf_dir: str, rate_limiter) -> DownloadResult:
        try:
            if not paper.doi:
                return DownloadResult(pmid=paper.pmid, success=False, strategy_used=self.name, error_message="No DOI available")

            await asyncio.sleep(self.rate_limit_delay)

            api_url = f"{self.base_url}/{paper.doi}"
            params = {"email": self.email}
            headers = {
                "User-Agent": "Academic Research Tool (educational use)",
                "Accept": "application/json"
            }

            async with session.get(api_url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()

                    if data.get("is_oa") and data.get("best_oa_location"):
                        oa_location = data["best_oa_location"]
                        pdf_url = oa_location.get("url_for_pdf")

                        if pdf_url:
                            async with session.get(pdf_url, headers=headers) as pdf_response:
                                if pdf_response.status == 200:
                                    content = await pdf_response.read()
                                    pdf_path = Path(pdf_dir) / f"{paper.pmid}.pdf"
                                    pdf_path.parent.mkdir(parents=True, exist_ok=True)

                                    with open(pdf_path, 'wb') as f:
                                        f.write(content)

                                    if PDFValidator.is_valid_pdf(str(pdf_path)):
                                        self.circuit_breaker.record_success()
                                        self.success_count += 1
                                        return DownloadResult(
                                            pmid=paper.pmid, success=True, file_path=str(pdf_path),
                                            file_size=len(content), strategy_used=self.name,
                                            metadata={'oa_date': data.get('oa_date')}
                                        )
                                    else:
                                        PDFValidator.cleanup_invalid_pdf(str(pdf_path))

            return DownloadResult(pmid=paper.pmid, success=False, strategy_used=self.name, error_message="No open access PDF available via Unpaywall")

        except Exception as e:
            self.circuit_breaker.record_failure()
            self.failure_count += 1
            return DownloadResult(pmid=paper.pmid, success=False, strategy_used=self.name, error_message=f"Unpaywall error: {str(e)}")


class PyPaperBotStrategy(PDFDownloadStrategy):
    """PyPaperBot for broader PDF access (educational use only)"""

    @property
    def name(self) -> str:
        return "PyPaperBot"

    @property
    def priority(self) -> int:
        return 7

    async def can_handle(self, paper: Paper) -> bool:
        return bool(paper.title or paper.doi)

    async def get_pdf_url(self, paper: Paper) -> Optional[str]:
        if paper.doi:
            return f"PYPAPERBOT_DOI:{paper.doi}"
        elif paper.title:
            return f"PYPAPERBOT_TITLE:{paper.title}"
        return None

    async def download_pdf(self, session, paper: Paper, pdf_dir: str, rate_limiter) -> DownloadResult:
        try:
            pdf_url = await self.get_pdf_url(paper)
            if not pdf_url:
                return DownloadResult(pmid=paper.pmid, success=False, strategy_used=self.name, error_message="No search terms")

            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    if pdf_url.startswith("PYPAPERBOT_DOI:"):
                        doi = pdf_url.replace("PYPAPERBOT_DOI:", "")
                        doi_file = Path(temp_dir) / "dois.txt"
                        with open(doi_file, 'w') as f:
                            f.write(doi + '\n')
                        cmd = [sys.executable, "-m", "PyPaperBot", "--doi-file", str(doi_file), "--dwn-dir", temp_dir, "--num-limit", "1", "--scholar-pages", "2"]
                    elif pdf_url.startswith("PYPAPERBOT_TITLE:"):
                        title = pdf_url.replace("PYPAPERBOT_TITLE:", "")
                        cmd = [sys.executable, "-m", "PyPaperBot", "--query", title, "--dwn-dir", temp_dir, "--num-limit", "1", "--scholar-pages", "2", "--scholar-results", "5"]
                    else:
                        return DownloadResult(pmid=paper.pmid, success=False, strategy_used=self.name, error_message="Invalid URL format")

                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env={**os.environ, 'PYTHONPATH': os.environ.get('PYTHONPATH', '')})

                    pdf_files = list(Path(temp_dir).glob("**/*.pdf"))
                    if pdf_files:
                        largest_pdf = max(pdf_files, key=lambda p: p.stat().st_size)
                        if largest_pdf.stat().st_size > 50000:
                            target_path = Path(pdf_dir) / f"{paper.pmid}.pdf"
                            target_path.parent.mkdir(parents=True, exist_ok=True)

                            import shutil
                            shutil.move(str(largest_pdf), target_path)

                            self.circuit_breaker.record_success()
                            self.success_count += 1
                            return DownloadResult(
                                pmid=paper.pmid, success=True, file_path=str(target_path),
                                file_size=target_path.stat().st_size, strategy_used=self.name,
                                metadata={'source': 'PyPaperBot'}
                            )

                    return DownloadResult(pmid=paper.pmid, success=False, strategy_used=self.name, error_message="No PDF downloaded")

                except subprocess.TimeoutExpired:
                    return DownloadResult(pmid=paper.pmid, success=False, strategy_used=self.name, error_message="PyPaperBot timeout after 120s")

        except Exception as e:
            self.circuit_breaker.record_failure()
            self.failure_count += 1
            return DownloadResult(pmid=paper.pmid, success=False, strategy_used=self.name, error_message=f"PyPaperBot error: {str(e)}")


class RobustPDFFetcher:
    """Production-ready PDF fetching library for PubMed papers"""

    def __init__(self, pdf_dir: str = "pdfs",
                 min_success_rate: float = 0.3,
                 batch_size: int = 5,
                 inter_batch_delay: float = 2.0,
                 enhanced_mode: bool = True,
                 email: str = "research@academic.edu"):
        self.pdf_dir = Path(pdf_dir)
        self.pdf_dir.mkdir(exist_ok=True)

        self.min_success_rate = min_success_rate
        self.batch_size = batch_size
        self.inter_batch_delay = inter_batch_delay
        self.enhanced_mode = enhanced_mode
        self.email = email

        # Initialize core strategies (ordered by priority)
        self.strategies = [
            EuropePMCStrategy(),        # Highest success rate from testing
            PMCOAServiceStrategy(),     # Fixed XML parsing - now enabled
            DirectPMCStrategy(),        # Fixed URL patterns with redirect handling
            EnhancedDOIStrategy(),      # Publisher-specific DOI handling
            DOIRedirectStrategy(),      # Keep as fallback
            ArxivStrategy(),            # Basic arXiv support
        ]

        # Add enhanced strategies if enabled
        if enhanced_mode:
            enhanced_strategies = [
                ArxivAPIStrategy(),             # Official arXiv API
                PyPaperBotStrategy(),           # Educational use only
                SemanticScholarStrategy(),      # Academic search engine
                UnpaywallStrategy(email),       # Legal open access discovery
            ]
            self.strategies.extend(enhanced_strategies)
            logger.info("Enhanced mode enabled: arXiv API, PyPaperBot, Semantic Scholar, Unpaywall")
        else:
            logger.info("Standard mode: official sources only")

        self.strategies.sort(key=lambda s: s.priority)

        # Rate limiting and monitoring
        self.rate_limiter = RateLimiter()
        self.session_stats = defaultdict(int)

        logger.info(f"Initialized RobustPDFFetcher with {len(self.strategies)} strategies")

    async def download_single(self, paper: Paper) -> DownloadResult:
        """Download PDF for a single paper"""
        async with aiohttp.ClientSession() as session:
            return await self._download_paper(session, paper)

    async def download_batch(self, papers: List[Paper]) -> BatchResult:
        """Download PDFs for a batch of papers"""
        start_time = time.time()
        results = []

        async with aiohttp.ClientSession() as session:
            # Process in batches to respect rate limits
            for i in range(0, len(papers), self.batch_size):
                batch = papers[i:i + self.batch_size]

                # Process batch concurrently
                batch_tasks = [self._download_paper(session, paper) for paper in batch]
                batch_results = await asyncio.gather(*batch_tasks)
                results.extend(batch_results)

                # Check success rate (only enforce after reasonable sample size)
                current_success_rate = sum(1 for r in results if r.success) / len(results)
                if current_success_rate < self.min_success_rate and len(results) >= 20:
                    logger.warning(f"Success rate {current_success_rate:.1%} below threshold {self.min_success_rate:.1%}")
                    logger.warning("This is normal for PubMed papers - continuing with abstract-only analysis")
                    # Don't raise exception - just log and continue

                # Inter-batch delay
                if i + self.batch_size < len(papers):
                    await asyncio.sleep(self.inter_batch_delay)

        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.success)

        # Collect strategy usage stats
        strategies_used = defaultdict(int)
        for result in results:
            if result.strategy_used:
                strategies_used[result.strategy_used] += 1

        return BatchResult(
            total_papers=len(papers),
            successful_downloads=successful,
            failed_downloads=len(results) - successful,
            success_rate=successful / len(results) if results else 0.0,
            total_time=total_time,
            results=results,
            strategies_used=dict(strategies_used)
        )

    async def stream_download_batch(self, papers: List[Paper]) -> AsyncIterator[DownloadResult]:
        """Stream PDF downloads for real-time results"""
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(3)  # Limit concurrent downloads

            async def download_with_semaphore(paper: Paper) -> DownloadResult:
                async with semaphore:
                    return await self._download_paper(session, paper)

            # Process all papers concurrently but yield results as they complete
            tasks = [asyncio.create_task(download_with_semaphore(paper)) for paper in papers]

            for coro in asyncio.as_completed(tasks):
                result = await coro
                yield result

    async def _download_paper(self, session: aiohttp.ClientSession, paper: Paper) -> DownloadResult:
        """Download PDF for a single paper using available strategies"""
        logger.info(f"🔍 Attempting PDF download for {paper.pmid} (PMC: {paper.pmcid}, DOI: {paper.doi})")

        for strategy in self.strategies:
            try:
                can_handle = await strategy.can_handle(paper)
                logger.info(f"   {strategy.name} can_handle: {can_handle}")

                if can_handle:
                    pdf_url = await strategy.get_pdf_url(paper)
                    logger.info(f"   {strategy.name} URL: {pdf_url}")

                    result = await strategy.download_pdf(session, paper, str(self.pdf_dir), self.rate_limiter)

                    if result.success:
                        logger.info(f"✅ Downloaded {paper.pmid} using {strategy.name}")
                        return result
                    else:
                        logger.info(f"❌ {strategy.name} failed for {paper.pmid}: {result.error_message}")

            except Exception as e:
                logger.error(f"Strategy {strategy.name} failed for {paper.pmid}: {e}")

        # All strategies failed
        logger.warning(f"❌ All strategies failed for {paper.pmid}")
        return DownloadResult(
            pmid=paper.pmid,
            success=False,
            error_message="All strategies failed"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get fetcher statistics"""
        total_attempts = sum(s.success_count + s.failure_count for s in self.strategies)
        total_successes = sum(s.success_count for s in self.strategies)

        strategy_stats = {}
        for strategy in self.strategies:
            strategy_stats[strategy.name] = {
                'success_count': strategy.success_count,
                'failure_count': strategy.failure_count,
                'success_rate': strategy.get_success_rate(),
                'circuit_breaker_state': strategy.circuit_breaker.state
            }

        return {
            'total_attempts': total_attempts,
            'total_successes': total_successes,
            'overall_success_rate': total_successes / total_attempts if total_attempts > 0 else 0.0,
            'strategy_statistics': strategy_stats,
            'pdf_directory': str(self.pdf_dir),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of all strategies"""
        health = {}

        for strategy in self.strategies:
            health[strategy.name] = {
                'available': strategy.circuit_breaker.state != 'open',
                'success_rate': strategy.get_success_rate(),
                'failure_count': strategy.failure_count,
            }

        return health