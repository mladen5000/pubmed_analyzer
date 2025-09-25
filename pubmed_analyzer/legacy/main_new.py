#!/usr/bin/env python3
"""
Modular Scientific Literature Analysis Pipeline
Combines full-text PDF processing, RAG, vector search, and advanced analytics
"""

import os
import sys
import asyncio
import argparse
import logging
from typing import List, Optional

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import our modular components
from pubmed_analyzer.models.paper import Paper
from pubmed_analyzer.core.search import PubMedSearcher
from pubmed_analyzer.core.id_converter import PMIDToPMCConverter
from pubmed_analyzer.core.pdf_fetcher import UnifiedPDFFetcher


class ModularPubMedPipeline:
    """Modular PubMed analysis pipeline with unified PDF fetching"""

    def __init__(
        self,
        email: str,
        api_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        deepseek_key: Optional[str] = None
    ):
        self.email = email
        self.api_key = api_key
        self.openai_key = openai_key
        self.deepseek_key = deepseek_key

        # Initialize components
        self.searcher = PubMedSearcher(email, api_key)
        self.id_converter = PMIDToPMCConverter(email, api_key)
        self.pdf_fetcher = UnifiedPDFFetcher()

        logger.info(f"Initialized modular pipeline with email: {email}")
        if api_key:
            logger.info("Using NCBI API key - higher rate limits enabled")

    async def run_pipeline(
        self,
        query: str,
        max_papers: int = 50,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Paper]:
        """
        Run the complete analysis pipeline

        Args:
            query: PubMed search query
            max_papers: Maximum number of papers to analyze
            start_date: Start date filter (YYYY/MM/DD)
            end_date: End date filter (YYYY/MM/DD)

        Returns:
            List of Paper objects with metadata and content
        """
        logger.info(f"🔍 Starting pipeline for query: '{query}'")

        # Step 1: Search PubMed
        logger.info("Step 1: Searching PubMed...")
        pmids = await self.searcher.search_papers(
            query=query,
            max_results=max_papers,
            start_date=start_date,
            end_date=end_date
        )

        if not pmids:
            logger.warning("No papers found for the given query")
            return []

        # Step 2: Fetch metadata
        logger.info("Step 2: Fetching paper metadata...")
        papers = await self.searcher.fetch_papers_metadata(pmids)

        # Step 3: Convert to PMC IDs and check OA availability
        logger.info("Step 3: Converting PMIDs to PMC IDs and checking Open Access...")
        await self.id_converter.enrich_with_pmcids(papers)

        # Count papers with full-text potential
        fulltext_papers = [p for p in papers if p.has_fulltext]
        logger.info(f"Found {len(fulltext_papers)}/{len(papers)} papers with potential full-text access")

        # Step 4: Download PDFs
        logger.info("Step 4: Downloading PDFs using unified fetcher...")
        await self.pdf_fetcher.download_all(papers)

        # Report final statistics
        successful_downloads = sum(1 for p in papers if p.download_success)
        logger.info(f"✅ Pipeline complete: {successful_downloads} PDFs downloaded successfully")

        return papers

    def get_analysis_summary(self, papers: List[Paper]) -> dict:
        """Generate summary statistics for the analysis"""
        total_papers = len(papers)
        with_pmcids = sum(1 for p in papers if p.pmcid)
        with_fulltext_potential = sum(1 for p in papers if p.has_fulltext)
        downloaded_pdfs = sum(1 for p in papers if p.download_success)
        extracted_text = sum(1 for p in papers if p.full_text)

        return {
            'total_papers': total_papers,
            'with_pmcids': with_pmcids,
            'with_fulltext_potential': with_fulltext_potential,
            'downloaded_pdfs': downloaded_pdfs,
            'extracted_text': extracted_text,
            'success_rates': {
                'pmcid_conversion': (with_pmcids / total_papers) * 100 if total_papers > 0 else 0,
                'pdf_download': (downloaded_pdfs / with_fulltext_potential) * 100 if with_fulltext_potential > 0 else 0,
                'text_extraction': (extracted_text / downloaded_pdfs) * 100 if downloaded_pdfs > 0 else 0,
            }
        }


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Modular PubMed Literature Analysis Pipeline"
    )

    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="PubMed search query"
    )

    parser.add_argument(
        "--max-papers", "-n",
        type=int,
        default=50,
        help="Maximum number of papers to analyze (default: 50)"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date filter (YYYY/MM/DD format)"
    )

    parser.add_argument(
        "--end-date",
        type=str,
        help="End date filter (YYYY/MM/DD format)"
    )

    parser.add_argument(
        "--email", "-e",
        type=str,
        help="Email address for NCBI API (overrides default)"
    )

    return parser.parse_args()


async def main():
    """Main execution function"""
    args = parse_arguments()

    # Configuration - CHANGE THE EMAIL ADDRESS TO YOUR OWN
    EMAIL = args.email if args.email else "mrasic2@uic.edu"
    NCBI_API_KEY = os.getenv("NCBI_API_KEY")
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")

    # Validate email
    if EMAIL == "mrasic2@uic.edu" and not args.email:
        logger.warning("⚠️  Using default email address. Please change it in main.py or use --email flag")
        logger.warning("⚠️  NCBI requires a valid email address for API access")

    # Initialize pipeline
    pipeline = ModularPubMedPipeline(
        email=EMAIL,
        api_key=NCBI_API_KEY,
        openai_key=OPENAI_KEY,
        deepseek_key=DEEPSEEK_KEY
    )

    try:
        # Run the analysis pipeline
        papers = await pipeline.run_pipeline(
            query=args.query,
            max_papers=args.max_papers,
            start_date=args.start_date,
            end_date=args.end_date
        )

        if not papers:
            logger.error("No papers found or processed")
            return

        # Generate summary
        summary = pipeline.get_analysis_summary(papers)

        logger.info("📊 Analysis Summary:")
        logger.info(f"   Total papers: {summary['total_papers']}")
        logger.info(f"   PMC IDs found: {summary['with_pmcids']} ({summary['success_rates']['pmcid_conversion']:.1f}%)")
        logger.info(f"   PDFs downloaded: {summary['downloaded_pdfs']} ({summary['success_rates']['pdf_download']:.1f}%)")
        logger.info(f"   Text extracted: {summary['extracted_text']} ({summary['success_rates']['text_extraction']:.1f}%)")

        # TODO: Integrate the analysis components from the original main.py
        # For now, we have successfully modularized the data collection pipeline
        logger.info("🎉 Modular pipeline execution complete!")
        logger.info("📝 Analysis components (topic modeling, sentiment analysis, etc.) to be integrated next")

        # Save basic results for now
        import json
        results = {
            'summary': summary,
            'papers': [
                {
                    'pmid': p.pmid,
                    'pmcid': p.pmcid,
                    'title': p.title,
                    'journal': p.journal,
                    'has_fulltext': p.has_fulltext,
                    'download_success': p.download_success,
                    'pdf_path': p.pdf_path,
                    'error_message': p.error_message,
                }
                for p in papers
            ]
        }

        with open("modular_pipeline_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("Results saved to modular_pipeline_results.json")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())