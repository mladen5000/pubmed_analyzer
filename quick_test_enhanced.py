#!/usr/bin/env python3
"""
Quick test of enhanced PDF fetcher - single paper test
"""

import asyncio
import logging
import os
from pathlib import Path

from pubmed_analyzer.api.pdf_fetcher_api import PubMedPDFFetcher

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def quick_test():
    """Quick test with a single paper"""

    email = os.getenv('PUBMED_EMAIL', 'test@example.com')

    # Create test directory
    test_dir = Path("quick_test_pdfs")
    test_dir.mkdir(exist_ok=True)

    logger.info("Quick Enhanced PDF Fetcher Test")

    # Test enhanced mode
    fetcher = PubMedPDFFetcher(
        email=email,
        pdf_dir=str(test_dir),
        enhanced_mode=True,
        batch_size=2
    )

    # Test with just 2 papers
    result = await fetcher.download_from_search(
        query="COVID-19 vaccine efficacy",
        max_results=2
    )

    logger.info(f"Results:")
    logger.info(f"  Total papers: {result.total_papers}")
    logger.info(f"  Successful downloads: {result.successful_downloads}")
    logger.info(f"  Success rate: {result.success_rate:.1%}")
    logger.info(f"  Strategies used: {result.strategies_used}")

    # Check files
    pdf_files = list(test_dir.glob("*.pdf"))
    logger.info(f"  Downloaded files: {len(pdf_files)}")

    # Get strategy info
    try:
        stats = fetcher.get_statistics()
        if 'strategy_info' in stats:
            logger.info("Strategy info available ✓")
            for name, info in stats['strategy_info'].items():
                logger.info(f"  {name}: {info.get('success_count', 0)} successes")
        else:
            logger.info("Basic stats only")
    except Exception as e:
        logger.error(f"Stats error: {e}")

    return result


if __name__ == "__main__":
    try:
        result = asyncio.run(quick_test())
        print(f"\nTest completed! Success rate: {result.success_rate:.1%}")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()