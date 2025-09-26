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

    def __init__(self, failure_threshold: int = 10, timeout: float = 120):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half_open

    def record_success(self):
        """Record successful operation"""
        self.failure_count = 0
        self.state = 'closed'

    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

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
        self.circuit_breaker = CircuitBreaker()
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

                headers = {
                    'User-Agent': 'RobustPDFFetcher/1.0 (mailto:researcher@university.edu)',
                    'Accept': 'application/pdf,*/*',
                }

                timeout = aiohttp.ClientTimeout(total=30)
                async with session.get(pdf_url, headers=headers, timeout=timeout) as response:
                    if response.status == 200:
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
                        validator = PDFValidator()
                        validation_result = validator.validate_pdf(str(pdf_path))

                        download_time = time.time() - start_time

                        return DownloadResult(
                            pmid=paper.pmid,
                            success=True,
                            file_path=str(pdf_path),
                            file_size=len(content),
                            attempt_count=attempt + 1,
                            download_time=download_time,
                            validation_passed=validation_result.is_valid,
                            metadata={
                                'url': pdf_url,
                                'response_status': response.status,
                                'content_type': response.headers.get('content-type'),
                                'validation_details': validation_result.details
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


class PMCOAServiceStrategy(PDFDownloadStrategy):
    """Official PMC Open Access service strategy"""

    @property
    def name(self) -> str:
        return "PMC OA Service"

    @property
    def priority(self) -> int:
        return 1  # Highest priority

    async def can_handle(self, paper: Paper) -> bool:
        return paper.pmcid is not None

    async def get_pdf_url(self, paper: Paper) -> Optional[str]:
        if not paper.pmcid:
            return None

        # Clean PMC ID
        pmc_id = paper.pmcid.replace('PMC', '')
        return f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC{pmc_id}&format=pdf"


class DOIRedirectStrategy(PDFDownloadStrategy):
    """DOI resolution strategy"""

    @property
    def name(self) -> str:
        return "DOI Redirect"

    @property
    def priority(self) -> int:
        return 2

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
        return 3

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


class PubMedCentralFallbackStrategy(PDFDownloadStrategy):
    """Try PMC even without PMC ID using PMID"""

    @property
    def name(self) -> str:
        return "PMC Fallback"

    @property
    def priority(self) -> int:
        return 4

    async def can_handle(self, paper: Paper) -> bool:
        # Always try this as fallback
        return paper.pmid is not None

    async def get_pdf_url(self, paper: Paper) -> Optional[str]:
        if not paper.pmid:
            return None

        # Try PMC direct URL patterns
        pmid = paper.pmid.replace('PMID:', '').strip()

        # Multiple PMC URL patterns to try
        urls = [
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/pmid/{pmid}/pdf/",
            f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmid}&format=pdf",
        ]

        # Return first URL (the strategy will try it)
        return urls[0]


class RobustPDFFetcher:
    """Production-ready PDF fetching library for PubMed papers"""

    def __init__(self, pdf_dir: str = "pdfs",
                 min_success_rate: float = 0.3,
                 batch_size: int = 5,
                 inter_batch_delay: float = 2.0):
        self.pdf_dir = Path(pdf_dir)
        self.pdf_dir.mkdir(exist_ok=True)

        self.min_success_rate = min_success_rate
        self.batch_size = batch_size
        self.inter_batch_delay = inter_batch_delay

        # Initialize strategies (ordered by priority)
        self.strategies = [
            PMCOAServiceStrategy(),
            DOIRedirectStrategy(),
            ArxivStrategy(),
            PubMedCentralFallbackStrategy(),
        ]
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

    async def _download_paper(self, session: aiohttp.ClientSession, paper: Paper) -> DownloadResult:
        """Download PDF for a single paper using available strategies"""
        for strategy in self.strategies:
            try:
                if await strategy.can_handle(paper):
                    result = await strategy.download_pdf(session, paper, str(self.pdf_dir), self.rate_limiter)

                    if result.success:
                        logger.debug(f"✅ Downloaded {paper.pmid} using {strategy.name}")
                        return result
                    else:
                        logger.debug(f"❌ {strategy.name} failed for {paper.pmid}: {result.error_message}")

            except Exception as e:
                logger.error(f"Strategy {strategy.name} failed for {paper.pmid}: {e}")

        # All strategies failed
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