#!/usr/bin/env python3
"""
Enhanced Multi-Source PDF Fetcher
Extends robust PDF fetcher with third-party sources: arXiv API, paperscraper, PyPaperBot
"""

import asyncio
import logging
import tempfile
import subprocess
import sys
from typing import Optional, Dict, Any, List
from pathlib import Path

from .robust_pdf_fetcher import PDFDownloadStrategy, DownloadResult, RobustPDFFetcher
from ..models.paper import Paper
from .advanced_pdf_strategies import SemanticScholarStrategy, COREStrategy, UnpaywallStrategy

logger = logging.getLogger(__name__)


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


class PaperscraperStrategy(PDFDownloadStrategy):
    """Paperscraper for preprint servers (arXiv, bioRxiv, medRxiv)"""

    @property
    def name(self) -> str:
        return "Paperscraper"

    @property
    def priority(self) -> int:
        return 6

    async def can_handle(self, paper: Paper) -> bool:
        if not paper.title:
            return False
        title_lower = str(paper.title).lower()
        return any(keyword in title_lower for keyword in ['arxiv', 'biorxiv', 'medrxiv', 'preprint'])

    async def get_pdf_url(self, paper: Paper) -> Optional[str]:
        if paper.doi and 'arxiv' in paper.doi.lower():
            arxiv_id = paper.doi.split('/')[-1].replace('arxiv.', '')
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        elif paper.title:
            return f"PAPERSCRAPER_SEARCH:{paper.title}"
        return None

    async def download_pdf(self, session, paper: Paper, pdf_dir: str, rate_limiter) -> DownloadResult:
        try:
            import paperscraper
            from paperscraper.arxiv import get_and_dump_arxiv_papers
            import json

            pdf_url = await self.get_pdf_url(paper)
            if not pdf_url:
                return DownloadResult(pmid=paper.pmid, success=False, strategy_used=self.name, error_message="No search terms")

            if pdf_url.startswith("https://arxiv.org/pdf/"):
                return await super().download_pdf(session, paper, pdf_dir, rate_limiter)

            if pdf_url.startswith("PAPERSCRAPER_SEARCH:"):
                search_term = pdf_url.replace("PAPERSCRAPER_SEARCH:", "")

                with tempfile.TemporaryDirectory() as temp_dir:
                    try:
                        query = [[search_term]]
                        result_file = Path(temp_dir) / "results.jsonl"
                        get_and_dump_arxiv_papers(query, str(result_file))

                        if result_file.exists():
                            with open(result_file, 'r') as f:
                                results = [json.loads(line) for line in f]

                            if results:
                                best_match = self._find_best_match(search_term, results)
                                if best_match and 'pdf_url' in best_match:
                                    import requests
                                    response = requests.get(best_match['pdf_url'], timeout=30)
                                    if response.status_code == 200:
                                        target_path = Path(pdf_dir) / f"{paper.pmid}.pdf"
                                        target_path.parent.mkdir(parents=True, exist_ok=True)

                                        with open(target_path, 'wb') as f:
                                            f.write(response.content)

                                        self.circuit_breaker.record_success()
                                        self.success_count += 1
                                        return DownloadResult(
                                            pmid=paper.pmid, success=True, file_path=str(target_path),
                                            file_size=target_path.stat().st_size, strategy_used=self.name,
                                            metadata={'source': 'arxiv', 'matched_id': best_match.get('id')}
                                        )
                    except Exception as e:
                        logger.debug(f"Paperscraper search failed: {e}")

            self.circuit_breaker.record_failure()
            self.failure_count += 1
            return DownloadResult(pmid=paper.pmid, success=False, strategy_used=self.name, error_message="No matches found")

        except ImportError:
            return DownloadResult(pmid=paper.pmid, success=False, strategy_used=self.name, error_message="paperscraper not available")

    def _find_best_match(self, search_term: str, results: List[Dict]) -> Optional[Dict]:
        try:
            from difflib import SequenceMatcher
            search_clean = search_term.lower().strip()
            best_match, best_score = None, 0.0

            for result in results:
                title = result.get('title', '').lower().strip()
                if title:
                    score = SequenceMatcher(None, search_clean, title).ratio()
                    if score > best_score and score > 0.4:
                        best_score, best_match = score, result
            return best_match
        except:
            return results[0] if results else None


class PyPaperBotStrategy(PDFDownloadStrategy):
    """PyPaperBot for broader PDF access (use with caution - educational only)"""

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

                        cmd = [sys.executable, "-m", "PyPaperBot", "--doi-file", str(doi_file), "--dwn-dir", temp_dir, "--num-limit", "1"]

                    elif pdf_url.startswith("PYPAPERBOT_TITLE:"):
                        title = pdf_url.replace("PYPAPERBOT_TITLE:", "")
                        cmd = [sys.executable, "-m", "PyPaperBot", "--query", title, "--dwn-dir", temp_dir, "--num-limit", "1", "--scholar-pages", "1"]

                    else:
                        return DownloadResult(pmid=paper.pmid, success=False, strategy_used=self.name, error_message="Invalid URL format")

                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

                    pdf_files = list(Path(temp_dir).glob("**/*.pdf"))
                    if pdf_files:
                        target_path = Path(pdf_dir) / f"{paper.pmid}.pdf"
                        target_path.parent.mkdir(parents=True, exist_ok=True)

                        import shutil
                        shutil.move(str(pdf_files[0]), target_path)

                        self.circuit_breaker.record_success()
                        self.success_count += 1
                        return DownloadResult(
                            pmid=paper.pmid, success=True, file_path=str(target_path),
                            file_size=target_path.stat().st_size, strategy_used=self.name,
                            metadata={'source': 'PyPaperBot'}
                        )

                    self.circuit_breaker.record_failure()
                    self.failure_count += 1
                    return DownloadResult(pmid=paper.pmid, success=False, strategy_used=self.name, error_message="No PDF downloaded")

                except Exception as e:
                    self.circuit_breaker.record_failure()
                    self.failure_count += 1
                    return DownloadResult(pmid=paper.pmid, success=False, strategy_used=self.name, error_message=f"PyPaperBot error: {str(e)}")

        except (ImportError, FileNotFoundError):
            return DownloadResult(pmid=paper.pmid, success=False, strategy_used=self.name, error_message="PyPaperBot not available")


class EnhancedPDFFetcher(RobustPDFFetcher):
    """Enhanced PDF fetcher with third-party sources for 60-80% success rates"""

    def __init__(self, pdf_dir: str = "pdfs", min_success_rate: float = 0.3,
                 batch_size: int = 5, inter_batch_delay: float = 2.0, enable_third_party: bool = True):

        super().__init__(pdf_dir, min_success_rate, batch_size, inter_batch_delay)

        if enable_third_party:
            enhanced_strategies = [
                ArxivAPIStrategy(),
                PaperscraperStrategy(),
                PyPaperBotStrategy(),
                # New advanced strategies for non-open access content
                SemanticScholarStrategy(),
                COREStrategy(),
                UnpaywallStrategy(),
            ]
            self.strategies.extend(enhanced_strategies)
            self.strategies.sort(key=lambda s: s.priority)

            logger.info(f"Enhanced PDF fetcher initialized with {len(self.strategies)} strategies")
            logger.info("Third-party sources enabled: arXiv API, paperscraper, PyPaperBot")
            logger.info("Advanced sources enabled: Semantic Scholar, CORE.ac.uk, Unpaywall")
        else:
            logger.info("Enhanced PDF fetcher initialized with only official sources")

    def get_strategy_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available strategies"""
        info = {}
        descriptions = {
            'EuropePMC': 'Europe PMC open access repository - high success rate',
            'PMC OA Service': 'Official NCBI PMC Open Access service',
            'Direct PMC': 'Direct PMC PDF links with redirect handling',
            'DOI Redirect': 'Publisher PDF access via DOI resolution',
            'arXiv': 'arXiv preprint server direct links',
            'arXiv API': 'Official arXiv Python API - most reliable for arXiv papers',
            'Paperscraper': 'Multi-source preprint server access (arXiv, bioRxiv, etc.)',
            'PyPaperBot': 'Broad PDF access via multiple sources (educational use only)',
            'Semantic Scholar API': 'Academic search engine with direct PDF access to millions of papers',
            'CORE.ac.uk API': 'Global repository of 28M+ academic papers from institutional repositories',
            'Unpaywall API': 'Legal open access discovery service for finding free versions of paywalled papers'
        }

        for strategy in self.strategies:
            info[strategy.name] = {
                'priority': strategy.priority,
                'success_count': strategy.success_count,
                'failure_count': strategy.failure_count,
                'success_rate': strategy.get_success_rate(),
                'circuit_breaker_state': strategy.circuit_breaker.state,
                'description': descriptions.get(strategy.name, 'Custom PDF download strategy')
            }
        return info