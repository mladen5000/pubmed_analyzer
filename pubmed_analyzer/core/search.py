import xml.etree.ElementTree as ET
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from ..models.paper import Paper
from ..utils.ncbi_client import NCBIClient

logger = logging.getLogger(__name__)


class PubMedSearcher:
    """Search PubMed and fetch paper metadata using NCBI E-utilities"""

    def __init__(self, email: str, api_key: Optional[str] = None):
        self.email = email
        self.api_key = api_key

    async def search_papers(
        self,
        query: str,
        max_results: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[str]:
        """
        Search PubMed for papers matching query

        Args:
            query: PubMed search query
            max_results: Maximum number of results to return
            start_date: Start date filter (YYYY/MM/DD format)
            end_date: End date filter (YYYY/MM/DD format)

        Returns:
            List of PubMed IDs
        """
        async with NCBIClient(self.email, self.api_key) as client:
            # Add date range to query if specified
            search_query = query
            if start_date and end_date:
                search_query = f"{query} AND ({start_date}[PDAT]:{end_date}[PDAT])"

            search_result = await client.search_pubmed(
                query=search_query,
                retmax=max_results
            )

            pmids = search_result.get("idlist", [])
            count = search_result.get("count", "0")

            logger.info(f"Found {len(pmids)} papers (total available: {count}) for query: {query}")
            return pmids

    async def fetch_papers_metadata(self, pmids: List[str], batch_size: int = None) -> List[Paper]:
        """
        Fetch paper metadata from PubMed for multiple PMIDs with optimized batching

        Args:
            pmids: List of PubMed IDs
            batch_size: Number of papers to fetch in each batch (auto-optimized if None)

        Returns:
            List of Paper objects with metadata
        """
        if not pmids:
            return []

        # Optimize batch size based on API key availability and total papers
        if batch_size is None:
            if self.api_key:
                # With API key: larger batches, more aggressive
                batch_size = min(200, max(50, len(pmids) // 5))
            else:
                # Without API key: smaller batches, more conservative
                batch_size = min(100, max(20, len(pmids) // 10))

        logger.info(f"Using optimized batch size: {batch_size} (API key: {'✅' if self.api_key else '❌'})")

        papers = []

        async with NCBIClient(self.email, self.api_key) as client:
            # Process in batches to stay within NCBI limits
            for i in tqdm(range(0, len(pmids), batch_size), desc="Fetching metadata"):
                batch = pmids[i : i + batch_size]

                try:
                    response = await client.fetch_metadata(batch, retmode='xml')
                    xml_content = await response.text()

                    # Parse XML response
                    batch_papers = self._parse_pubmed_xml(xml_content)
                    papers.extend(batch_papers)

                    # Add small delay between batches for courtesy
                    if i + batch_size < len(pmids):
                        delay = 0.5 if self.api_key else 1.0
                        await asyncio.sleep(delay)

                except Exception as e:
                    logger.error(f"Failed to fetch metadata for batch {i}-{i+len(batch)}: {e}")
                    # Create minimal Paper objects for failed fetches
                    for pmid in batch:
                        papers.append(Paper(pmid=pmid, error_message=str(e)))

        logger.info(f"Successfully fetched metadata for {len(papers)} papers")
        return papers

    def _parse_pubmed_xml(self, xml_content: str) -> List[Paper]:
        """
        Parse PubMed XML response into Paper objects

        Args:
            xml_content: XML response from NCBI efetch

        Returns:
            List of Paper objects
        """
        papers = []

        try:
            root = ET.fromstring(xml_content)

            for article in root.findall(".//PubmedArticle"):
                try:
                    paper = self._parse_single_article(article)
                    papers.append(paper)
                except Exception as e:
                    logger.error(f"Failed to parse individual article: {e}")

        except ET.ParseError as e:
            logger.error(f"Failed to parse PubMed XML: {e}")

        return papers

    def _parse_single_article(self, article: ET.Element) -> Paper:
        """
        Parse a single PubmedArticle XML element

        Args:
            article: PubmedArticle XML element

        Returns:
            Paper object with parsed metadata
        """
        # Extract PMID
        pmid_elem = article.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None else ""

        # Extract basic article info
        citation = article.find(".//Article")
        if citation is None:
            return Paper(pmid=pmid, error_message="No article citation found")

        # Title
        title_elem = citation.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None else None

        # Abstract
        abstract_parts = []
        for abstract_elem in citation.findall(".//AbstractText"):
            text = abstract_elem.text or ""
            # Handle structured abstracts
            label = abstract_elem.get("Label", "")
            if label:
                text = f"{label}: {text}"
            abstract_parts.append(text)

        abstract = " ".join(abstract_parts) if abstract_parts else None

        # Authors
        authors = []
        author_list = citation.find(".//AuthorList")
        if author_list is not None:
            for author in author_list.findall("Author"):
                last_name = author.find("LastName")
                fore_name = author.find("ForeName")

                if last_name is not None and fore_name is not None:
                    authors.append(f"{fore_name.text} {last_name.text}")
                elif last_name is not None:
                    authors.append(last_name.text)

        # Journal
        journal_elem = citation.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else None

        # Publication date
        pub_date = self._parse_publication_date(citation)

        # DOI
        doi = None
        for article_id in article.findall(".//ArticleId"):
            if article_id.get("IdType") == "doi":
                doi = article_id.text
                break

        return Paper(
            pmid=pmid,
            title=title,
            abstract=abstract,
            authors=authors,
            journal=journal,
            pub_date=pub_date,
            doi=doi
        )

    def _parse_publication_date(self, citation: ET.Element) -> Optional[datetime]:
        """
        Parse publication date from XML

        Args:
            citation: Article XML element

        Returns:
            datetime object or None if parsing fails
        """
        try:
            # Try multiple date formats
            date_elements = [
                citation.find(".//PubDate"),
                citation.find(".//ArticleDate"),
            ]

            for date_elem in date_elements:
                if date_elem is None:
                    continue

                year_elem = date_elem.find("Year")
                month_elem = date_elem.find("Month")
                day_elem = date_elem.find("Day")

                if year_elem is not None:
                    year = int(year_elem.text)
                    month = 1
                    day = 1

                    if month_elem is not None:
                        try:
                            month = int(month_elem.text)
                        except ValueError:
                            # Handle month names
                            month_names = {
                                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
                                'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                                'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                            }
                            month = month_names.get(month_elem.text, 1)

                    if day_elem is not None:
                        try:
                            day = int(day_elem.text)
                        except ValueError:
                            day = 1

                    return datetime(year, month, day)

        except Exception as e:
            logger.debug(f"Failed to parse publication date: {e}")

        return None