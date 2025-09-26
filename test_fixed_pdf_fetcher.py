#!/usr/bin/env python3
"""
Test script to verify PDF fetching fixes
"""

import asyncio
import logging
from pubmed_analyzer.models.paper import Paper
from pubmed_analyzer.core.robust_pdf_fetcher import RobustPDFFetcher

# Set up logging
logging.basicConfig(level=logging.INFO)

async def test_fixed_fetcher():
    """Test the fixed PDF fetcher with known working PMC IDs"""
    print("🧪 Testing Fixed PDF Fetcher")
    print("=" * 50)

    # Create test papers with known PMC IDs that should work
    test_papers = [
        Paper(pmid="32526867", pmcid="PMC8443998", title="Test Paper 1"),
        Paper(pmid="31247177", pmcid="PMC6557568", title="Test Paper 2", doi="10.1371/journal.pone.0218004"),
        Paper(pmid="32641130", pmcid="PMC7308628", title="Test Paper 3"),
    ]

    print(f"Testing {len(test_papers)} papers with known PMC IDs...")

    # Initialize fetcher
    fetcher = RobustPDFFetcher("test_fixed_pdfs")

    # Run download
    batch_result = await fetcher.download_batch(test_papers)

    print("\n📊 RESULTS:")
    print(f"Success Rate: {batch_result.success_rate:.1%}")
    print(f"Downloaded: {batch_result.successful_downloads}/{batch_result.total_papers}")
    print(f"Total Time: {batch_result.total_time:.1f}s")

    print(f"\nStrategy Usage: {batch_result.strategies_used}")

    print("\n📄 Individual Results:")
    for result in batch_result.results:
        status = "✅" if result.success else "❌"
        strategy = result.strategy_used or "None"
        error = result.error_message or "Success"
        print(f"{status} {result.pmid}: {strategy} - {error}")

        if result.success:
            print(f"   📁 Saved to: {result.file_path}")
            print(f"   📏 Size: {result.file_size} bytes")

    return batch_result.success_rate

if __name__ == "__main__":
    success_rate = asyncio.run(test_fixed_fetcher())

    if success_rate > 0.5:
        print(f"\n🎉 SUCCESS! Fixed fetcher achieved {success_rate:.1%} success rate")
    else:
        print(f"\n🚨 Still issues - only {success_rate:.1%} success rate")