#!/usr/bin/env python3
"""
PubMed Analyzer CLI
Simple interface with two modes: 'abstracts' and 'full'
"""

import os
import sys
import asyncio
import argparse
import logging
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import components
from pubmed_analyzer.core.search import PubMedSearcher
from pubmed_analyzer.core.id_converter import PMIDToPMCConverter
from pubmed_analyzer.api.pdf_fetcher_api import PubMedPDFFetcher

try:
    from pubmed_analyzer.utils.abstract_visualizer import AbstractOptimizedVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError:
    logger.warning("Visualizer not available")
    VISUALIZER_AVAILABLE = False


async def run_abstracts_mode(args):
    """
    ABSTRACTS MODE: Ultra-fast abstract-only analysis
    - No PMC conversion overhead
    - No PDF downloads
    - Rich visualizations from abstracts
    - Perfect for quick analysis and exploration
    """
    logger.info("🚀 ABSTRACTS MODE: Ultra-fast analysis")
    start_time = datetime.now()

    # Initialize searcher only
    searcher = PubMedSearcher(args.email, args.api_key)

    # Step 1: Search PubMed
    logger.info(f"🔍 Searching PubMed for: '{args.query}'")
    pmids = await searcher.search_papers(
        query=args.query,
        max_results=args.max_papers,
        start_date=args.start_date,
        end_date=args.end_date
    )

    if not pmids:
        logger.error("No papers found for the query")
        return

    # Step 2: Fetch metadata with abstracts
    logger.info("📄 Fetching paper metadata and abstracts...")
    papers = await searcher.fetch_papers_metadata(pmids)

    # Report abstract coverage
    with_abstracts = sum(1 for p in papers if p.abstract and p.abstract.strip())
    logger.info(f"✅ Retrieved {with_abstracts}/{len(papers)} papers with abstracts ({with_abstracts/len(papers)*100:.1f}%)")

    # Step 3: Generate visualizations (if enabled)
    if args.visualizations and VISUALIZER_AVAILABLE:
        logger.info("📊 Generating abstract-optimized visualizations...")
        visualizer = AbstractOptimizedVisualizer()

        # Create results structure expected by visualizer
        results = {
            "papers": papers,
            "query": args.query,
            "pipeline_summary": {
                "total_papers": len(papers),
                "with_abstracts": with_abstracts,
                "mode": "abstract_only"
            }
        }

        visualization_files = visualizer.create_abstract_dashboard(results, args.query)
        logger.info(f"✅ Generated {len(visualization_files)} visualizations")

        for viz_file in visualization_files:
            logger.info(f"   📈 {viz_file}")

    # Report summary
    elapsed = datetime.now() - start_time
    logger.info(f"🎉 ABSTRACTS MODE complete in {elapsed.total_seconds():.1f} seconds!")
    logger.info(f"   📚 Papers analyzed: {len(papers)}")
    logger.info(f"   📄 Abstract coverage: {with_abstracts/len(papers)*100:.1f}%")
    if args.visualizations and VISUALIZER_AVAILABLE:
        logger.info(f"   📊 Visualizations: {len(visualization_files)} files")


async def run_streaming_full_mode(args):
    """
    STREAMING FULL MODE: Start downloading PDFs as soon as papers are processed
    - Pipeline processing: PMID → Metadata → PMC ID → PDF Download
    - Faster time-to-first-PDF
    - Progressive results
    - Better resource utilization
    """
    logger.info("🚀 STREAMING FULL MODE: Pipeline processing for faster results")
    logger.info("💡 PDFs will start downloading immediately as papers are processed")
    start_time = datetime.now()

    # Initialize searcher and converter
    searcher = PubMedSearcher(args.email, args.api_key)
    converter = PMIDToPMCConverter(args.email, args.api_key)

    # Initialize PDF fetcher with streaming capability
    pdf_fetcher = PubMedPDFFetcher(
        email=args.email,
        api_key=args.api_key,
        pdf_dir=args.pdf_dir or "pdfs",
        enhanced_mode=not args.no_enhanced
    )

    logger.info(f"🔍 Starting streaming search: '{args.query}'")

    # Step 1: Search for PMIDs
    pmids = await searcher.search_papers(
        query=args.query,
        max_results=args.max_papers,
        start_date=args.start_date,
        end_date=args.end_date
    )

    if not pmids:
        logger.error("No papers found for the query")
        return

    # Step 2: Fetch metadata
    logger.info(f"📄 Fetching metadata for {len(pmids)} papers...")
    papers = await searcher.fetch_papers_metadata(pmids)

    # Step 3: Convert PMC IDs
    logger.info("🔗 Converting PMIDs to PMC IDs...")
    await converter.enrich_with_pmcids(papers)

    # Step 4: Stream download PDFs
    logger.info("📚 Starting streaming PDF downloads...")
    results = []
    async for result in pdf_fetcher.pdf_fetcher.stream_download_batch(papers):
        results.append(result)

        # Show real-time progress for successful downloads
        if result.success:
            logger.info(f"✅ Downloaded {result.pmid} using {result.strategy_used}")

        # Show progress every 5 papers
        if len(results) % 5 == 0:
            successful = sum(1 for r in results if r.success)
            logger.info(f"📋 Progress: {successful}/{len(results)} PDFs downloaded ({successful/len(results)*100:.1f}% success)")

    # Final summary
    elapsed = datetime.now() - start_time
    successful = sum(1 for r in results if r.success)

    # Collect strategy usage stats
    strategies_used = {}
    for result in results:
        if result.strategy_used:
            strategies_used[result.strategy_used] = strategies_used.get(result.strategy_used, 0) + 1

    logger.info(f"🎉 STREAMING FULL MODE complete in {elapsed.total_seconds():.1f} seconds!")
    logger.info(f"   📚 Papers processed: {len(results)}")
    logger.info(f"   📄 PDFs downloaded: {successful}")
    logger.info(f"   📈 Success rate: {successful/len(results)*100:.1f}%")

    # Strategy breakdown
    if strategies_used:
        logger.info("   🔧 Strategies used:")
        for strategy, count in strategies_used.items():
            logger.info(f"      {strategy}: {count} downloads")

    # Show failed downloads
    failed_results = [r for r in results if not r.success]
    if failed_results:
        logger.info(f"   ❌ Failed downloads: {len(failed_results)}")
        for i, failed in enumerate(failed_results[:3]):
            logger.info(f"      {failed.pmid}: {failed.error_message}")
        if len(failed_results) > 3:
            logger.info(f"      ... and {len(failed_results) - 3} more")


async def run_full_mode(args):
    """
    FULL MODE: Comprehensive analysis with robust PDF downloading
    - Multi-strategy PDF downloading
    - Rate limiting and error handling
    - Batch processing with success monitoring
    - Comprehensive analysis with full-text when available
    """
    if args.streaming:
        return await run_streaming_full_mode(args)

    logger.info("📚 FULL MODE: Comprehensive analysis with PDF downloading")
    logger.info("💡 Note: Most PubMed papers don't have freely accessible PDFs (expect 20-40% success rate)")
    start_time = datetime.now()

    # Initialize robust PDF fetcher
    pdf_fetcher = PubMedPDFFetcher(
        email=args.email,
        api_key=args.api_key,
        pdf_dir=args.pdf_dir or "pdfs",
        min_success_rate=0.1,  # More realistic threshold for PDF downloads (10%)
        batch_size=5
    )

    # Step 1: Search PubMed
    logger.info(f"🔍 Searching PubMed for: '{args.query}'")
    searcher = PubMedSearcher(args.email, args.api_key)

    pmids = await searcher.search_papers(
        query=args.query,
        max_results=args.max_papers,
        start_date=args.start_date,
        end_date=args.end_date
    )

    if not pmids:
        logger.error("No papers found for the query")
        return

    # Step 2: Fetch metadata
    logger.info("📄 Fetching paper metadata...")
    papers = await searcher.fetch_papers_metadata(pmids)

    # Step 3: Convert PMIDs to PMC IDs (needed for PDF downloading)
    logger.info("🔗 Converting PMIDs to PMC IDs...")
    converter = PMIDToPMCConverter(args.email, args.api_key)
    await converter.enrich_with_pmcids(papers)

    pmcid_count = sum(1 for p in papers if p.pmcid)
    logger.info(f"✅ Found PMC IDs for {pmcid_count}/{len(papers)} papers")

    # DEBUG: Show sample of papers and their data
    logger.info("🔍 DEBUG: Sample paper data:")
    for i, paper in enumerate(papers[:3]):
        logger.info(f"   Paper {i+1}: PMID={paper.pmid}, PMC={paper.pmcid}, DOI={paper.doi}")
        logger.info(f"           Title: {paper.title[:50] if paper.title else 'None'}...")

    if pmcid_count == 0:
        logger.warning("⚠️  No PMC IDs found - most strategies will be skipped!")

    # Step 4: Download PDFs
    logger.info("📚 Starting robust PDF downloads...")
    result = await pdf_fetcher.download_from_papers(papers)

    # Step 3: Generate visualizations (if enabled)
    if args.visualizations and VISUALIZER_AVAILABLE:
        logger.info("📊 Generating comprehensive visualizations...")
        # You could integrate with your existing enhanced visualizer here
        # For now, just generate abstract-optimized ones
        visualizer = AbstractOptimizedVisualizer()

        # Create papers list from results for visualizer
        papers = []  # You'd need to reconstruct Paper objects from results if needed

        logger.info("📊 Visualization integration pending - using PDF download results for now")

    # Report comprehensive summary
    elapsed = datetime.now() - start_time
    logger.info(f"🎉 FULL MODE complete in {elapsed.total_seconds():.1f} seconds!")
    logger.info(f"   📚 Papers found: {result.total_papers}")
    logger.info(f"   📄 PDFs downloaded: {result.successful_downloads}")
    logger.info(f"   📈 Success rate: {result.success_rate:.1%}")
    logger.info(f"   ⚡ Download time: {result.total_time:.1f}s")

    # Strategy breakdown
    if result.strategies_used:
        logger.info("   🔧 Strategies used:")
        for strategy, count in result.strategies_used.items():
            logger.info(f"      {strategy}: {count} downloads")

    # Individual results summary
    failed_results = [r for r in result.results if not r.success]
    if failed_results:
        logger.info(f"   ❌ Failed downloads: {len(failed_results)}")
        # Show first few failure reasons
        for i, failed in enumerate(failed_results[:3]):
            logger.info(f"      {failed.pmid}: {failed.error_message}")
        if len(failed_results) > 3:
            logger.info(f"      ... and {len(failed_results) - 3} more")

    return result


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="PubMed Analyzer: Choose 'abstracts' for fast analysis or 'full' for PDF downloads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pubmed_analyzer.py abstracts --query "machine learning" --max-papers 100
  python pubmed_analyzer.py full --query "COVID-19" --max-papers 50 --pdf-dir covid_pdfs
  python pubmed_analyzer.py abstracts --query "CRISPR" --visualizations --max-papers 200
        """
    )

    # Subcommands for mode selection
    subparsers = parser.add_subparsers(dest='mode', help='Analysis mode')

    # ABSTRACTS mode
    abstracts_parser = subparsers.add_parser(
        'abstracts',
        help='Ultra-fast abstract-only analysis (recommended for exploration)'
    )
    abstracts_parser.add_argument(
        '--query', '-q', required=True, help='PubMed search query'
    )
    abstracts_parser.add_argument(
        '--max-papers', '-n', type=int, default=100,
        help='Maximum papers to analyze (default: 100)'
    )
    abstracts_parser.add_argument(
        '--visualizations', '-v', action='store_true',
        help='Generate visualizations from abstracts'
    )
    abstracts_parser.add_argument(
        '--start-date', help='Start date filter (YYYY/MM/DD)'
    )
    abstracts_parser.add_argument(
        '--end-date', help='End date filter (YYYY/MM/DD)'
    )

    # FULL mode
    full_parser = subparsers.add_parser(
        'full',
        help='Comprehensive analysis with robust PDF downloading'
    )
    full_parser.add_argument(
        '--query', '-q', required=True, help='PubMed search query'
    )
    full_parser.add_argument(
        '--max-papers', '-n', type=int, default=50,
        help='Maximum papers to download (default: 50)'
    )
    full_parser.add_argument(
        '--pdf-dir', help='Directory to save PDFs (default: pdfs)'
    )
    full_parser.add_argument(
        '--visualizations', '-v', action='store_true',
        help='Generate comprehensive visualizations'
    )
    full_parser.add_argument(
        '--start-date', help='Start date filter (YYYY/MM/DD)'
    )
    full_parser.add_argument(
        '--end-date', help='End date filter (YYYY/MM/DD)'
    )
    full_parser.add_argument(
        '--streaming', '-s', action='store_true',
        help='Use streaming mode for faster time-to-first-PDF (experimental)'
    )
    full_parser.add_argument(
        '--no-enhanced', action='store_true',
        help='Disable enhanced PDF strategies (official sources only)'
    )

    # Common arguments
    for subparser in [abstracts_parser, full_parser]:
        subparser.add_argument(
            '--email', '-e',
            default=os.getenv('PUBMED_EMAIL', 'mrasic2@uic.edu'),
            help='Email for NCBI API (default: from PUBMED_EMAIL env var)'
        )
        subparser.add_argument(
            '--api-key',
            default=os.getenv('NCBI_API_KEY'),
            help='NCBI API key for higher rate limits (default: from NCBI_API_KEY env var)'
        )

    args = parser.parse_args()

    # Validate mode selection
    if not args.mode:
        parser.error("Please specify a mode: 'abstracts' or 'full'")

    # Validate email
    if not args.email or args.email == 'mrasic2@uic.edu':
        logger.warning("⚠️  Using default email. Set PUBMED_EMAIL env var or use --email")
        logger.warning("⚠️  NCBI requires a valid email for API access")

    # Show API key status
    if args.api_key:
        logger.info(f"✅ Using NCBI API key (higher rate limits enabled)")
    else:
        logger.info(f"💡 No API key - consider setting NCBI_API_KEY env var for better performance")

    # Run the appropriate mode
    try:
        if args.mode == 'abstracts':
            asyncio.run(run_abstracts_mode(args))
        elif args.mode == 'full':
            asyncio.run(run_full_mode(args))
    except KeyboardInterrupt:
        logger.info("\n⏹️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()