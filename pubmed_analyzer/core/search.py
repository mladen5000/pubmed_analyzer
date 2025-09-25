import xml.etree.ElementTree as ET
import logging
import asyncio
import re
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
        Fetch paper metadata from PubMed for multiple PMIDs with optimized batching and abstract recovery

        Args:
            pmids: List of PubMed IDs
            batch_size: Number of papers to fetch in each batch (auto-optimized if None)

        Returns:
            List of Paper objects with metadata and maximum abstract coverage
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
        pmids_needing_recovery = []

        async with NCBIClient(self.email, self.api_key) as client:
            # Phase 1: Primary metadata fetch with enhanced parameters
            for i in tqdm(range(0, len(pmids), batch_size), desc="Fetching metadata"):
                batch = pmids[i : i + batch_size]

                try:
                    # Use enhanced fetch with abstract focus
                    response = await client.fetch_metadata(batch, retmode='xml', rettype='abstract')
                    xml_content = await response.text()

                    # Parse XML response
                    batch_papers = self._parse_pubmed_xml(xml_content)
                    papers.extend(batch_papers)

                    # Track PMIDs without abstracts for recovery
                    for paper in batch_papers:
                        if not paper.abstract or paper.abstract.strip() == "":
                            pmids_needing_recovery.append(paper.pmid)

                    # Add small delay between batches for courtesy
                    if i + batch_size < len(pmids):
                        delay = 0.3 if self.api_key else 0.7  # Slightly faster for abstract focus
                        await asyncio.sleep(delay)

                except Exception as e:
                    logger.error(f"Failed to fetch metadata for batch {i}-{i+len(batch)}: {e}")
                    # Try fallback method for this batch
                    try:
                        response = await client.fetch_metadata_with_fallbacks(batch)
                        xml_content = await response.text()
                        batch_papers = self._parse_pubmed_xml(xml_content)
                        papers.extend(batch_papers)

                        # Still track missing abstracts
                        for paper in batch_papers:
                            if not paper.abstract or paper.abstract.strip() == "":
                                pmids_needing_recovery.append(paper.pmid)
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed for batch {i}-{i+len(batch)}: {fallback_error}")
                        # Create minimal Paper objects for failed fetches
                        for pmid in batch:
                            papers.append(Paper(pmid=pmid, error_message=str(e)))
                            pmids_needing_recovery.append(pmid)

            # Phase 2: Abstract recovery for missing abstracts
            initial_abstract_count = sum(1 for paper in papers if paper.abstract and paper.abstract.strip())
            logger.info(f"Initial abstract coverage: {initial_abstract_count}/{len(papers)} ({initial_abstract_count/len(papers)*100:.1f}%)")

            if pmids_needing_recovery:
                logger.info(f"Attempting to recover {len(pmids_needing_recovery)} missing abstracts")

                # Use specialized recovery methods
                recovered_abstracts = await client.fetch_missing_abstracts(pmids_needing_recovery)

                # Update papers with recovered abstracts
                recovered_count = 0
                for paper in papers:
                    if paper.pmid in recovered_abstracts:
                        paper.abstract = recovered_abstracts[paper.pmid]
                        recovered_count += 1

                final_abstract_count = sum(1 for paper in papers if paper.abstract and paper.abstract.strip())
                logger.info(f"Recovered {recovered_count} abstracts. Final coverage: {final_abstract_count}/{len(papers)} ({final_abstract_count/len(papers)*100:.1f}%)")

        logger.info(f"Successfully fetched metadata for {len(papers)} papers")
        return papers

    def get_abstract_coverage_stats(self, papers: List[Paper]) -> Dict[str, Any]:
        """
        Get detailed statistics about abstract coverage for analysis and optimization

        Args:
            papers: List of Paper objects

        Returns:
            Dict containing coverage statistics
        """
        total_papers = len(papers)
        papers_with_abstracts = sum(1 for paper in papers if paper.abstract and paper.abstract.strip())
        papers_without_abstracts = total_papers - papers_with_abstracts

        # Analyze abstract lengths
        abstract_lengths = [len(paper.abstract) for paper in papers if paper.abstract]
        avg_abstract_length = sum(abstract_lengths) / len(abstract_lengths) if abstract_lengths else 0

        # Identify common patterns in missing abstracts
        missing_patterns = {
            'error_messages': sum(1 for paper in papers if paper.error_message),
            'older_papers': 0,  # Could add date analysis
            'specific_journals': {},  # Could analyze by journal
        }

        return {
            'total_papers': total_papers,
            'papers_with_abstracts': papers_with_abstracts,
            'papers_without_abstracts': papers_without_abstracts,
            'coverage_percentage': (papers_with_abstracts / total_papers * 100) if total_papers > 0 else 0,
            'average_abstract_length': avg_abstract_length,
            'missing_patterns': missing_patterns,
            'optimization_suggestions': self._get_optimization_suggestions(papers)
        }

    def _get_optimization_suggestions(self, papers: List[Paper]) -> List[str]:
        """Generate suggestions for improving abstract coverage"""
        suggestions = []

        papers_without_abstracts = [p for p in papers if not p.abstract or not p.abstract.strip()]
        error_papers = [p for p in papers if p.error_message]

        if len(papers_without_abstracts) > 0:
            suggestions.append(f"Consider using fallback databases for {len(papers_without_abstracts)} papers without abstracts")

        if len(error_papers) > 0:
            suggestions.append(f"Retry failed requests for {len(error_papers)} papers with errors")

        # Check if we should use more aggressive recovery
        coverage = sum(1 for p in papers if p.abstract and p.abstract.strip()) / len(papers) * 100
        if coverage < 98:
            suggestions.append("Use individual PMID queries for remaining missing abstracts")

        return suggestions

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

        # Enhanced Abstract parsing - handles multiple formats and edge cases
        abstract_parts = []

        # Strategy 1: Look for AbstractText elements (most common)
        for abstract_elem in citation.findall(".//AbstractText"):
            text = abstract_elem.text or ""

            # Handle nested elements (like <i>, <b>, etc.) by getting all text
            if not text and len(abstract_elem):
                text = "".join(abstract_elem.itertext())

            if text.strip():
                # Handle structured abstracts with labels
                label = abstract_elem.get("Label", "")
                if label:
                    text = f"{label}: {text}"
                abstract_parts.append(text.strip())

        # Strategy 2: If no AbstractText, try Abstract element directly
        if not abstract_parts:
            abstract_elem = citation.find(".//Abstract")
            if abstract_elem is not None:
                # Get all text including nested elements
                text = "".join(abstract_elem.itertext()).strip()
                if text:
                    abstract_parts.append(text)

        # Strategy 3: Look for OtherAbstract (for translated abstracts, author abstracts)
        if not abstract_parts:
            for other_abstract in citation.findall(".//OtherAbstract"):
                text = "".join(other_abstract.itertext()).strip()
                if text:
                    abstract_type = other_abstract.get("Type", "Other")
                    abstract_parts.append(f"[{abstract_type}] {text}")

        # Strategy 4: Check Article element directly for any abstract content
        if not abstract_parts:
            # Sometimes abstracts are in unexpected locations
            article_text = "".join(citation.itertext())
            # Look for common abstract patterns
            abstract_patterns = [
                r'Abstract[:\s]+(.*?)(?:Keywords|Introduction|Background|Methods|PMID|$)',
                r'Summary[:\s]+(.*?)(?:Keywords|Introduction|Background|Methods|PMID|$)',
            ]
            for pattern in abstract_patterns:
                match = re.search(pattern, article_text, re.IGNORECASE | re.DOTALL)
                if match and len(match.group(1).strip()) > 50:  # Reasonable abstract length
                    abstract_parts.append(match.group(1).strip()[:1000])  # Limit length
                    break

        # Clean and combine abstract parts
        if abstract_parts:
            abstract = " ".join(abstract_parts)
            # Clean up common XML artifacts
            abstract = re.sub(r'\s+', ' ', abstract)  # Normalize whitespace
            abstract = abstract.strip()

            # Validate abstract quality
            if len(abstract) < 20:  # Too short to be meaningful
                abstract = None
        else:
            abstract = None

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