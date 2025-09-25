import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


class NCBIClient:
    """NCBI-compliant HTTP client with proper rate limiting and error handling"""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    PMC_OA_URL = "http://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

    def __init__(self, email: str, api_key: Optional[str] = None, tool: str = "pubmed_analyzer"):
        self.email = email
        self.api_key = api_key
        self.tool = tool

        # NCBI rate limiting: 3/sec without API key, 10/sec with key
        self.rate_limit = 10 if api_key else 3
        self.semaphore = asyncio.Semaphore(self.rate_limit)

        # Enhanced rate limiting tracking
        self.request_count = 0
        self.last_request_time = 0
        self.backoff_delay = 0.5  # Start with 500ms backoff for better safety
        self.consecutive_rate_limits = 0  # Track consecutive rate limit hits

        # Session will be created when needed
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry"""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': f'{self.tool}/1.0 ({self.email})'}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._session:
            await self._session.close()

    @property
    def session(self) -> aiohttp.ClientSession:
        """Get the current session or raise error if not initialized"""
        if self._session is None:
            raise RuntimeError("NCBIClient must be used as async context manager")
        return self._session

    async def make_request(self, endpoint: str, params: Dict[str, Any], retry_count: int = 0) -> aiohttp.ClientResponse:
        """
        Make a rate-limited request to NCBI E-utilities with intelligent backoff

        Args:
            endpoint: E-utilities endpoint (e.g., 'esearch.fcgi', 'efetch.fcgi')
            params: Query parameters
            retry_count: Current retry attempt (for exponential backoff)

        Returns:
            aiohttp.ClientResponse object
        """
        async with self.semaphore:
            # Add required NCBI parameters
            params.update({
                'email': self.email,
                'tool': self.tool
            })

            if self.api_key:
                params['api_key'] = self.api_key

            url = self.BASE_URL + endpoint

            # Adaptive delay based on request history
            current_time = asyncio.get_event_loop().time()
            time_since_last = current_time - self.last_request_time

            # Ensure minimum spacing between requests
            min_interval = 1.0 / self.rate_limit
            if time_since_last < min_interval:
                await asyncio.sleep(min_interval - time_since_last)

            try:
                response = await self.session.get(url, params=params)
                self.last_request_time = asyncio.get_event_loop().time()
                self.request_count += 1

                # Handle rate limiting with exponential backoff
                if response.status == 429:
                    self.consecutive_rate_limits += 1
                    if retry_count < 3:  # Max 3 retries
                        # More aggressive backoff based on consecutive hits
                        base_backoff = self.backoff_delay * (2 ** retry_count)
                        penalty_multiplier = min(5.0, 1.0 + (self.consecutive_rate_limits * 0.5))
                        backoff_time = base_backoff * penalty_multiplier

                        logger.warning(f"NCBI rate limit hit (attempt {retry_count + 1}/3), backing off {backoff_time:.2f}s (consecutive: {self.consecutive_rate_limits})")
                        await asyncio.sleep(backoff_time)
                        return await self.make_request(endpoint, params, retry_count + 1)
                    else:
                        logger.error("NCBI rate limit exceeded after 3 retries")
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=429,
                            message="Rate limit exceeded after retries"
                        )

                # Handle other temporary errors
                elif response.status in [502, 503, 504]:
                    if retry_count < 2:
                        backoff_time = 2.0 * (retry_count + 1)
                        logger.warning(f"NCBI server error {response.status}, retrying in {backoff_time}s")
                        await asyncio.sleep(backoff_time)
                        return await self.make_request(endpoint, params, retry_count + 1)

                response.raise_for_status()

                # Reset backoff on success
                self.consecutive_rate_limits = max(0, self.consecutive_rate_limits - 1)  # Gradual recovery
                self.backoff_delay = max(0.2, self.backoff_delay * 0.9)  # Gradually reduce backoff

                # Conservative delay - be extra respectful
                base_delay = 1.0 / self.rate_limit
                adaptive_delay = base_delay * (1.5 if not self.api_key else 1.0)  # 50% slower without API key
                await asyncio.sleep(adaptive_delay)

                return response

            except aiohttp.ClientError as e:
                logger.error(f"NCBI request failed for {endpoint}: {e}")
                raise

    async def make_pmc_request(self, params: Dict[str, Any]) -> aiohttp.ClientResponse:
        """
        Make request to PMC OA service

        Args:
            params: Query parameters for PMC OA service

        Returns:
            aiohttp.ClientResponse object
        """
        async with self.semaphore:
            try:
                response = await self.session.get(self.PMC_OA_URL, params=params)
                response.raise_for_status()

                # Be respectful to PMC servers
                await asyncio.sleep(0.5)
                return response

            except aiohttp.ClientError as e:
                logger.error(f"PMC OA request failed: {e}")
                raise

    async def search_pubmed(self, query: str, retmax: int = 50, retstart: int = 0) -> Dict[str, Any]:
        """
        Search PubMed using E-utilities esearch

        Args:
            query: PubMed search query
            retmax: Maximum number of results to return
            retstart: Starting index for pagination

        Returns:
            Parsed search results
        """
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': str(retmax),
            'retstart': str(retstart),
            'retmode': 'json'
        }

        response = await self.make_request('esearch.fcgi', params)
        data = await response.json()

        return data.get('esearchresult', {})

    async def fetch_metadata(self, pmids: list, retmode: str = 'xml') -> aiohttp.ClientResponse:
        """
        Fetch metadata for multiple PMIDs using efetch

        Args:
            pmids: List of PubMed IDs
            retmode: Return format ('xml', 'json', etc.)

        Returns:
            Response containing metadata
        """
        if not pmids:
            raise ValueError("No PMIDs provided")

        # Batch up to 200 IDs per NCBI recommendation
        if len(pmids) > 200:
            logger.warning(f"Truncating {len(pmids)} PMIDs to 200 for single request")
            pmids = pmids[:200]

        params = {
            'db': 'pubmed',
            'id': ','.join(str(pmid) for pmid in pmids),
            'retmode': retmode
        }

        return await self.make_request('efetch.fcgi', params)

    async def convert_pmid_to_pmcid(self, pmid: str) -> Optional[str]:
        """
        Convert PMID to PMC ID using elink

        Args:
            pmid: PubMed ID

        Returns:
            PMC ID if available, None otherwise
        """
        params = {
            'dbfrom': 'pubmed',
            'db': 'pmc',
            'id': pmid,
            'linkname': 'pubmed_pmc',
            'retmode': 'json'
        }

        try:
            response = await self.make_request('elink.fcgi', params)
            data = await response.json()

            linksets = data.get('linksets', [])
            if linksets and 'linksetdbs' in linksets[0]:
                linksetdbs = linksets[0]['linksetdbs']
                if linksetdbs and 'links' in linksetdbs[0]:
                    links = linksetdbs[0]['links']
                    if links:
                        return f"PMC{links[0]}"

            return None

        except Exception as e:
            logger.error(f"Failed to convert PMID {pmid} to PMC ID: {e}")
            return None