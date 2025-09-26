#!/usr/bin/env python3
"""
Test script for Enhanced PDF Fetcher
Tests the new multi-source PDF fetching capabilities
"""

import asyncio
import logging
import os
from pathlib import Path

from pubmed_analyzer.api.pdf_fetcher_api import PubMedPDFFetcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_enhanced_pdf_fetcher():
    """Test the enhanced PDF fetcher with various paper types"""

    # Get email from environment or use default
    email = os.getenv('PUBMED_EMAIL', 'test@example.com')
    api_key = os.getenv('NCBI_API_KEY')

    # Create test directory
    test_dir = Path("test_enhanced_pdfs")
    test_dir.mkdir(exist_ok=True)

    logger.info("Testing Enhanced PDF Fetcher")
    logger.info(f"Email: {email}")
    logger.info(f"API Key: {'✓' if api_key else '✗'}")
    logger.info(f"Test directory: {test_dir}")

    # Initialize enhanced fetcher
    fetcher = PubMedPDFFetcher(
        email=email,
        api_key=api_key,
        pdf_dir=str(test_dir),
        enhanced_mode=True  # Enable third-party sources
    )

    # Test with different types of papers
    test_cases = [
        {
            "name": "arXiv Paper (should work with arXiv API)",
            "query": "arXiv machine learning transformers",
            "max_results": 2
        },
        {
            "name": "Recent COVID-19 Papers (might include preprints)",
            "query": "COVID-19 2023",
            "max_results": 3
        },
        {
            "name": "Open Access Papers (should work with PMC)",
            "query": 'COVID-19 AND "open access"',
            "max_results": 3
        }
    ]

    all_results = []

    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Test Case {i}: {test_case['name']}")
        logger.info(f"Query: {test_case['query']}")
        logger.info(f"{'='*60}")

        try:
            result = await fetcher.download_from_search(
                query=test_case['query'],
                max_results=test_case['max_results']
            )

            all_results.append({
                'test_case': test_case['name'],
                'result': result
            })

            logger.info(f"Results for '{test_case['name']}':")
            logger.info(f"  Total papers: {result.total_papers}")
            logger.info(f"  Successful downloads: {result.successful_downloads}")
            logger.info(f"  Success rate: {result.success_rate:.1%}")
            logger.info(f"  Total time: {result.total_time:.1f}s")

            if result.strategies_used:
                logger.info("  Strategies used:")
                for strategy, count in result.strategies_used.items():
                    logger.info(f"    {strategy}: {count}")

            # Show individual results
            for res in result.results:
                status = "✅" if res.success else "❌"
                strategy = res.strategy_used or "N/A"
                logger.info(f"    {status} PMID {res.pmid} via {strategy}")
                if not res.success:
                    logger.info(f"      Error: {res.error_message}")

        except Exception as e:
            logger.error(f"Test case failed: {e}")
            all_results.append({
                'test_case': test_case['name'],
                'error': str(e)
            })

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("ENHANCED PDF FETCHER TEST SUMMARY")
    logger.info(f"{'='*60}")

    total_papers = sum(r['result'].total_papers for r in all_results if 'result' in r)
    total_successful = sum(r['result'].successful_downloads for r in all_results if 'result' in r)
    overall_success_rate = total_successful / total_papers if total_papers > 0 else 0

    logger.info(f"Total papers tested: {total_papers}")
    logger.info(f"Total successful downloads: {total_successful}")
    logger.info(f"Overall success rate: {overall_success_rate:.1%}")

    # Strategy statistics
    logger.info("\nStrategy Performance:")
    try:
        stats = fetcher.get_statistics()
        if 'strategy_info' in stats:
            for strategy_name, info in stats['strategy_info'].items():
                success_count = info.get('success_count', 0)
                failure_count = info.get('failure_count', 0)
                success_rate = info.get('success_rate', 0)
                logger.info(f"  {strategy_name}: {success_count} success, {failure_count} failures ({success_rate:.1%})")
    except Exception as e:
        logger.warning(f"Could not get strategy statistics: {e}")

    # Check downloaded files
    pdf_files = list(test_dir.glob("*.pdf"))
    logger.info(f"\nDownloaded files: {len(pdf_files)}")
    for pdf_file in pdf_files:
        size_mb = pdf_file.stat().st_size / (1024 * 1024)
        logger.info(f"  {pdf_file.name}: {size_mb:.1f} MB")

    logger.info(f"\nTest completed. Files saved to: {test_dir}")

    return all_results


async def test_standard_vs_enhanced():
    """Compare standard vs enhanced mode"""

    email = os.getenv('PUBMED_EMAIL', 'test@example.com')
    api_key = os.getenv('NCBI_API_KEY')

    test_query = "machine learning COVID-19"
    max_results = 5

    logger.info(f"\n{'='*60}")
    logger.info("COMPARING STANDARD vs ENHANCED MODE")
    logger.info(f"{'='*60}")
    logger.info(f"Query: {test_query}")
    logger.info(f"Max results: {max_results}")

    # Test standard mode
    logger.info("\n--- Testing STANDARD mode (official sources only) ---")
    standard_fetcher = PubMedPDFFetcher(
        email=email,
        api_key=api_key,
        pdf_dir="test_standard_pdfs",
        enhanced_mode=False
    )

    standard_result = await standard_fetcher.download_from_search(test_query, max_results)

    # Test enhanced mode
    logger.info("\n--- Testing ENHANCED mode (with third-party sources) ---")
    enhanced_fetcher = PubMedPDFFetcher(
        email=email,
        api_key=api_key,
        pdf_dir="test_enhanced_comparison_pdfs",
        enhanced_mode=True
    )

    enhanced_result = await enhanced_fetcher.download_from_search(test_query, max_results)

    # Compare results
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON RESULTS")
    logger.info(f"{'='*60}")

    logger.info(f"Standard Mode:")
    logger.info(f"  Success rate: {standard_result.success_rate:.1%}")
    logger.info(f"  Successful downloads: {standard_result.successful_downloads}/{standard_result.total_papers}")
    logger.info(f"  Strategies used: {standard_result.strategies_used}")

    logger.info(f"\nEnhanced Mode:")
    logger.info(f"  Success rate: {enhanced_result.success_rate:.1%}")
    logger.info(f"  Successful downloads: {enhanced_result.successful_downloads}/{enhanced_result.total_papers}")
    logger.info(f"  Strategies used: {enhanced_result.strategies_used}")

    improvement = enhanced_result.success_rate - standard_result.success_rate
    logger.info(f"\nImprovement: {improvement:.1%}")

    if improvement > 0:
        logger.info("✅ Enhanced mode showed improvement!")
    elif improvement == 0:
        logger.info("🟡 No difference between modes")
    else:
        logger.info("❌ Standard mode performed better")


if __name__ == "__main__":
    print("Enhanced PDF Fetcher Test Suite")
    print("=" * 50)

    # Check email configuration
    if not os.getenv('PUBMED_EMAIL'):
        print("⚠️  WARNING: PUBMED_EMAIL not set. Using test email.")
        print("   Set your email: export PUBMED_EMAIL='your.email@example.com'")

    if not os.getenv('NCBI_API_KEY'):
        print("ℹ️  INFO: NCBI_API_KEY not set. Using default rate limits.")
        print("   Get API key at: https://www.ncbi.nlm.nih.gov/account/settings/")

    print("\nStarting tests...")

    try:
        # Run main test
        asyncio.run(test_enhanced_pdf_fetcher())

        print("\n" + "="*50)
        print("Running comparison test...")

        # Run comparison test
        asyncio.run(test_standard_vs_enhanced())

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nTest suite completed!")