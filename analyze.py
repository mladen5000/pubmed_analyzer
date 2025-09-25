#!/usr/bin/env python3
"""
PubMed Literature Analysis Pipeline
Main entry point for the modular scientific literature analysis system.
"""

import os
import asyncio
import argparse
import logging
from typing import List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,  # Back to INFO for normal operation
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import our modular components
from pubmed_analyzer import (
    Paper,
    PubMedSearcher,
    PMIDToPMCConverter,
    UnifiedPDFFetcher
)


class PubMedAnalysisPipeline:
    """Complete PubMed analysis pipeline with modular architecture"""

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

        # Initialize pipeline components
        self.searcher = PubMedSearcher(email, api_key)
        self.id_converter = PMIDToPMCConverter(email, api_key)
        self.pdf_fetcher = UnifiedPDFFetcher()

        logger.info(f"🚀 Initialized PubMed Analysis Pipeline")
        logger.info(f"   Email: {email}")
        if api_key:
            logger.info("   NCBI API key: ✅ (higher rate limits enabled)")
        else:
            logger.info("   NCBI API key: ❌ (using default rate limits)")

    async def analyze(
        self,
        query: str,
        max_papers: int = 50,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Paper]:
        """
        Run complete literature analysis pipeline

        Args:
            query: PubMed search query
            max_papers: Maximum number of papers to analyze
            start_date: Start date filter (YYYY/MM/DD)
            end_date: End date filter (YYYY/MM/DD)

        Returns:
            List of analyzed Paper objects
        """
        logger.info(f"🔍 Starting analysis for query: '{query}'")
        logger.info(f"   Max papers: {max_papers}")
        if start_date or end_date:
            logger.info(f"   Date range: {start_date or 'earliest'} to {end_date or 'latest'}")

        # Phase 1: Paper Discovery
        logger.info("\n📚 Phase 1: Paper Discovery")
        pmids = await self.searcher.search_papers(
            query=query,
            max_results=max_papers,
            start_date=start_date,
            end_date=end_date
        )

        if not pmids:
            logger.warning("❌ No papers found for the given query")
            return []

        # Phase 2: Metadata Collection
        logger.info("\n📝 Phase 2: Metadata Collection")
        papers = await self.searcher.fetch_papers_metadata(pmids)
        logger.info(f"✅ Collected metadata for {len(papers)} papers")

        # Phase 3: Full-text Discovery
        logger.info("\n🔓 Phase 3: Full-text Discovery")
        await self.id_converter.enrich_with_pmcids(papers)

        fulltext_papers = [p for p in papers if p.has_fulltext]
        logger.info(f"✅ Found {len(fulltext_papers)}/{len(papers)} papers with full-text access")

        # Phase 4: PDF Collection
        logger.info("\n📄 Phase 4: PDF Collection")
        await self.pdf_fetcher.download_all(papers)

        successful_downloads = sum(1 for p in papers if p.download_success)
        logger.info(f"✅ Successfully downloaded {successful_downloads} PDFs")

        # Phase 5: Analysis Summary
        logger.info("\n📊 Phase 5: Analysis Summary")
        self._report_results(papers)

        return papers

    def _report_results(self, papers: List[Paper]) -> None:
        """Generate and display analysis summary"""
        total = len(papers)
        with_pmcids = sum(1 for p in papers if p.pmcid)
        with_fulltext = sum(1 for p in papers if p.has_fulltext)
        downloaded = sum(1 for p in papers if p.download_success)
        extracted_text = sum(1 for p in papers if p.full_text)
        converted_markdown = sum(1 for p in papers if p.markdown_path)

        logger.info("=" * 60)
        logger.info("📊 ANALYSIS RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total papers found:        {total}")
        logger.info(f"PMC IDs discovered:        {with_pmcids} ({(with_pmcids/total)*100 if total > 0 else 0:.1f}%)")
        logger.info(f"Open access papers:        {with_fulltext} ({(with_fulltext/total)*100 if total > 0 else 0:.1f}%)")
        logger.info(f"PDFs downloaded:           {downloaded} ({(downloaded/with_fulltext)*100 if with_fulltext > 0 else 0:.1f}% of available)")
        logger.info(f"Text extracted:            {extracted_text} ({(extracted_text/downloaded)*100 if downloaded > 0 else 0:.1f}% of PDFs)")
        logger.info(f"Markdown converted:        {converted_markdown} ({(converted_markdown/downloaded)*100 if downloaded > 0 else 0:.1f}% of PDFs)")

        if downloaded > 0:
            total_size = sum(os.path.getsize(p.pdf_path) for p in papers
                           if p.pdf_path and os.path.exists(p.pdf_path))
            logger.info(f"Total PDF size:            {total_size / (1024*1024):.1f} MB")

        logger.info("=" * 60)

        # Save detailed results
        self._save_results(papers)

    def _save_results(self, papers: List[Paper]) -> None:
        """Save analysis results to JSON file"""
        import json

        results = {
            'analysis_metadata': {
                'total_papers': len(papers),
                'papers_with_pmcids': sum(1 for p in papers if p.pmcid),
                'papers_with_fulltext': sum(1 for p in papers if p.has_fulltext),
                'pdfs_downloaded': sum(1 for p in papers if p.download_success),
                'text_extracted': sum(1 for p in papers if p.full_text),
                'markdown_converted': sum(1 for p in papers if p.markdown_path)
            },
            'papers': [
                {
                    'pmid': p.pmid,
                    'pmcid': p.pmcid,
                    'title': p.title,
                    'authors': p.authors,
                    'journal': p.journal,
                    'pub_date': p.pub_date.isoformat() if p.pub_date else None,
                    'doi': p.doi,
                    'has_fulltext': p.has_fulltext,
                    'download_success': p.download_success,
                    'pdf_path': p.pdf_path,
                    'txt_path': p.txt_path,
                    'markdown_path': p.markdown_path,
                    'error_message': p.error_message,
                    'pmc_metadata': p.pmc_metadata
                }
                for p in papers
            ]
        }

        output_file = "pubmed_analysis_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"📁 Detailed results saved to: {output_file}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="PubMed Literature Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --query "COVID-19 treatment"
  %(prog)s --query "machine learning medicine" --max-papers 100
  %(prog)s --query "cancer therapy" --start-date 2020/01/01 --end-date 2023/12/31
  %(prog)s --query "free full text[sb]" --email your@email.com
        """
    )

    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="PubMed search query (use PubMed advanced search syntax)"
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
        help="Email address for NCBI API (required by NCBI)"
    )

    return parser.parse_args()


async def main():
    """Main execution function"""
    args = parse_arguments()

    # Configuration
    EMAIL = args.email if args.email else "mrasic2@uic.edu"
    NCBI_API_KEY = os.getenv("NCBI_API_KEY")
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")

    # Email validation
    if EMAIL == "mrasic2@uic.edu" and not args.email:
        logger.warning("⚠️  Using default email - please change in code or use --email flag")
        logger.warning("⚠️  NCBI requires a valid email for API access compliance")

    try:
        # Initialize and run analysis
        pipeline = PubMedAnalysisPipeline(
            email=EMAIL,
            api_key=NCBI_API_KEY,
            openai_key=OPENAI_KEY,
            deepseek_key=DEEPSEEK_KEY
        )

        papers = await pipeline.analyze(
            query=args.query,
            max_papers=args.max_papers,
            start_date=args.start_date,
            end_date=args.end_date
        )

        if papers:
            logger.info(f"🎉 Analysis complete! Processed {len(papers)} papers")
        else:
            logger.error("❌ No papers were processed")

    except KeyboardInterrupt:
        logger.info("\n⏹️  Analysis interrupted by user")
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())