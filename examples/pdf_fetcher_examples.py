#!/usr/bin/env python3
"""
Usage Examples for Robust PDF Fetching Library
Demonstrates various ways to use the PubMed PDF fetching capabilities
"""

import asyncio
import os
from pathlib import Path

# Import the API
from pubmed_analyzer.api import PubMedPDFFetcher
from pubmed_analyzer.api.pdf_fetcher_api import (
    PubMedPDFFetcherSync,
    download_pdfs_by_pmids,
    download_pdfs_by_search,
    download_pdfs_sync
)


async def example_1_basic_async():
    """Example 1: Basic async usage"""
    print("=== Example 1: Basic Async Usage ===")

    # Initialize the fetcher
    fetcher = PubMedPDFFetcher(
        email="your.email@example.com",  # REQUIRED: Change to your email
        api_key=os.getenv("NCBI_API_KEY"),  # Optional but recommended
        pdf_dir="example_pdfs",
        min_success_rate=0.3,  # Stop if success rate drops below 30%
        batch_size=5
    )

    # Download by PMIDs
    pmids = ["33157158", "33157159", "33157160"]  # Example PMIDs
    result = await fetcher.download_from_pmids(pmids)

    print(f"Downloaded: {result.successful_downloads}/{result.total_papers}")
    print(f"Success rate: {result.success_rate:.1%}")
    print(f"Strategies used: {result.strategies_used}")

    # Print individual results
    for download_result in result.results:
        if download_result.success:
            print(f"✅ {download_result.pmid}: {download_result.file_path}")
            print(f"   Size: {download_result.file_size:,} bytes, "
                  f"Strategy: {download_result.strategy_used}")
        else:
            print(f"❌ {download_result.pmid}: {download_result.error_message}")


async def example_2_search_and_download():
    """Example 2: Search PubMed and download results"""
    print("\n=== Example 2: Search and Download ===")

    fetcher = PubMedPDFFetcher(
        email="your.email@example.com",
        api_key=os.getenv("NCBI_API_KEY"),
        pdf_dir="covid_pdfs"
    )

    # Search and download
    result = await fetcher.download_from_search(
        query="COVID-19 vaccine effectiveness",
        max_results=20,
        start_date="2023/01/01",
        end_date="2024/12/31"
    )

    print(f"Search and download complete:")
    print(f"Total papers: {result.total_papers}")
    print(f"Successful downloads: {result.successful_downloads}")
    print(f"Success rate: {result.success_rate:.1%}")
    print(f"Total time: {result.total_time:.1f} seconds")


def example_3_synchronous_usage():
    """Example 3: Synchronous usage (easier for simple scripts)"""
    print("\n=== Example 3: Synchronous Usage ===")

    # Synchronous wrapper
    fetcher = PubMedPDFFetcherSync(
        email="your.email@example.com",
        api_key=os.getenv("NCBI_API_KEY"),
        pdf_dir="sync_pdfs"
    )

    # Download single paper
    result = fetcher.download_single("33157158")
    if result.success:
        print(f"✅ Downloaded: {result.file_path}")
        print(f"File size: {result.file_size:,} bytes")
        print(f"Validation passed: {result.validation_passed}")
    else:
        print(f"❌ Download failed: {result.error_message}")

    # Batch download
    pmids = ["33157158", "33157159", "33157160"]
    batch_result = fetcher.download_from_pmids(pmids)
    print(f"Batch result: {batch_result.successful_downloads}/{batch_result.total_papers} successful")


async def example_4_convenience_functions():
    """Example 4: Using convenience functions"""
    print("\n=== Example 4: Convenience Functions ===")

    # Quick async download by PMIDs
    result = await download_pdfs_by_pmids(
        pmids=["33157158", "33157159"],
        email="your.email@example.com",
        api_key=os.getenv("NCBI_API_KEY"),
        pdf_dir="quick_pdfs"
    )
    print(f"Quick download: {result.success_rate:.1%} success rate")

    # Quick search and download
    result = await download_pdfs_by_search(
        query="machine learning bioinformatics",
        email="your.email@example.com",
        max_results=10,
        pdf_dir="ml_bio_pdfs"
    )
    print(f"Search download: {result.successful_downloads} PDFs downloaded")


def example_5_synchronous_convenience():
    """Example 5: Synchronous convenience function"""
    print("\n=== Example 5: Synchronous Convenience ===")

    # One-liner for synchronous download
    result = download_pdfs_sync(
        pmids=["33157158"],
        email="your.email@example.com",
        pdf_dir="oneliner_pdfs"
    )

    if result.successful_downloads > 0:
        print(f"✅ Quick sync download successful!")
        for r in result.results:
            if r.success:
                print(f"   File: {r.file_path}")
    else:
        print("❌ Quick sync download failed")


async def example_6_monitoring_and_statistics():
    """Example 6: Monitoring and statistics"""
    print("\n=== Example 6: Monitoring and Statistics ===")

    fetcher = PubMedPDFFetcher(
        email="your.email@example.com",
        api_key=os.getenv("NCBI_API_KEY"),
        pdf_dir="monitoring_pdfs"
    )

    # Download some papers
    await fetcher.download_from_pmids(["33157158", "33157159", "33157160"])

    # Get statistics
    stats = fetcher.get_statistics()
    print("📊 Download Statistics:")
    print(f"   Total attempts: {stats['total_attempts']}")
    print(f"   Total successes: {stats['total_successes']}")
    print(f"   Overall success rate: {stats['overall_success_rate']:.1%}")

    print("\n🔧 Strategy Performance:")
    for strategy, data in stats['strategy_statistics'].items():
        print(f"   {strategy}: {data['success_rate']:.1%} success rate "
              f"({data['success_count']}/{data['success_count'] + data['failure_count']} attempts)")

    # Health check
    health = await fetcher.health_check()
    print("\n🏥 Strategy Health Check:")
    for strategy, status in health.items():
        status_emoji = "✅" if status['available'] else "❌"
        print(f"   {status_emoji} {strategy}: {status['success_rate']:.1%} success rate")


async def example_7_error_handling():
    """Example 7: Error handling and edge cases"""
    print("\n=== Example 7: Error Handling ===")

    fetcher = PubMedPDFFetcher(
        email="your.email@example.com",
        api_key=os.getenv("NCBI_API_KEY"),
        pdf_dir="error_handling_pdfs",
        min_success_rate=0.1,  # Lower threshold for demo
    )

    # Try with some invalid PMIDs
    test_pmids = [
        "33157158",     # Valid PMID
        "invalid_pmid", # Invalid PMID
        "99999999999",  # Non-existent PMID
    ]

    result = await fetcher.download_from_pmids(test_pmids)

    print("🔍 Download Results with Errors:")
    for r in result.results:
        if r.success:
            print(f"   ✅ {r.pmid}: Successfully downloaded")
        else:
            print(f"   ❌ {r.pmid}: {r.error_message}")

    print(f"\nOverall: {result.successful_downloads}/{result.total_papers} successful")


async def main():
    """Run all examples"""
    print("🚀 Robust PDF Fetching Library Examples")
    print("=" * 50)

    # Note: You MUST change the email address before running
    email_warning = """
    ⚠️  IMPORTANT: You must change 'your.email@example.com' to your actual email
    address before running these examples. NCBI requires a valid email for API access.

    Also consider setting NCBI_API_KEY environment variable for higher rate limits.
    """
    print(email_warning)

    try:
        # Run examples (comment out as needed for testing)
        await example_1_basic_async()
        await example_2_search_and_download()
        example_3_synchronous_usage()
        await example_4_convenience_functions()
        example_5_synchronous_convenience()
        await example_6_monitoring_and_statistics()
        await example_7_error_handling()

        print("\n🎉 All examples completed!")

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("Make sure to update the email address and check your internet connection.")


if __name__ == "__main__":
    asyncio.run(main())