#!/usr/bin/env python3
"""
Test specific arXiv paper download
"""

import asyncio
import logging
import os
from pathlib import Path

from pubmed_analyzer.api.pdf_fetcher_api import PubMedPDFFetcher

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def test_arxiv_papers():
    """Test with papers that should have arXiv versions"""

    email = os.getenv('PUBMED_EMAIL', 'test@example.com')

    # Create test directory
    test_dir = Path("test_arxiv_pdfs")
    test_dir.mkdir(exist_ok=True)

    print("Testing arXiv paper download...")

    # Test enhanced mode
    fetcher = PubMedPDFFetcher(
        email=email,
        pdf_dir=str(test_dir),
        enhanced_mode=True,
        batch_size=1
    )

    # Search for papers with arXiv IDs
    result = await fetcher.download_from_search(
        query="arXiv machine learning neural networks",  # Should find arXiv papers
        max_results=2
    )

    print(f"Results:")
    print(f"  Total papers: {result.total_papers}")
    print(f"  Successful downloads: {result.successful_downloads}")
    print(f"  Success rate: {result.success_rate:.1%}")
    print(f"  Strategies used: {result.strategies_used}")

    # Show details
    for res in result.results:
        status = "✅" if res.success else "❌"
        strategy = res.strategy_used or "N/A"
        print(f"  {status} PMID {res.pmid} via {strategy}")
        if not res.success:
            print(f"    Error: {res.error_message}")

    # Check files
    pdf_files = list(test_dir.glob("*.pdf"))
    print(f"  Downloaded files: {len(pdf_files)}")

if __name__ == "__main__":
    asyncio.run(test_arxiv_papers())