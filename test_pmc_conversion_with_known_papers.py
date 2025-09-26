#!/usr/bin/env python3
"""
Test PMC ID conversion with papers that are known to have PMC IDs
"""

import asyncio
import logging
import os
import sys

# Add the pubmed_analyzer directory to the path
sys.path.insert(0, '/Users/mladenrasic/Projects/pubmed_analyzer')

from pubmed_analyzer.core.search import PubMedSearcher
from pubmed_analyzer.core.id_converter import PMIDToPMCConverter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_with_known_pmc_papers():
    """Test with papers that are known to have PMC IDs"""

    print("🔬 Testing PMC ID conversion with searches likely to have PMC papers...")

    # Use the same configuration as main_new.py
    EMAIL = "mrasic2@uic.edu"
    NCBI_API_KEY = os.getenv("NCBI_API_KEY")

    print(f"Using email: {EMAIL}")
    print(f"Using API key: {'Yes' if NCBI_API_KEY else 'No'}")

    searcher = PubMedSearcher(EMAIL, NCBI_API_KEY)
    id_converter = PMIDToPMCConverter(EMAIL, NCBI_API_KEY)

    # Test queries that are more likely to return open access papers
    test_queries = [
        'COVID-19 AND "open access" AND 2020:2021[pdat]',  # COVID papers from 2020-2021 with open access
        'CRISPR AND "PMC" AND 2020:2022[pdat]',            # CRISPR papers mentioning PMC
        'machine learning AND biomedical AND 2020:2022[pdat]',  # Recent ML papers
        'cancer AND genomics AND 2019:2021[pdat]',         # Cancer genomics papers
    ]

    for query in test_queries:
        print(f"\n📡 Testing query: '{query}'")

        try:
            # Search for papers
            pmids = await searcher.search_papers(query=query, max_results=10)
            print(f"Found {len(pmids)} PMIDs")

            if not pmids:
                print("❌ No papers found")
                continue

            # Fetch metadata
            papers = await searcher.fetch_papers_metadata(pmids)
            print(f"Retrieved metadata for {len(papers)} papers")

            # Convert PMC IDs
            await id_converter.enrich_with_pmcids(papers)

            # Report results
            pmcid_count = sum(1 for p in papers if p.pmcid)
            success_rate = (pmcid_count / len(papers)) * 100 if papers else 0

            print(f"✅ PMC conversion: {pmcid_count}/{len(papers)} ({success_rate:.1f}%)")

            # Show some examples
            if pmcid_count > 0:
                print("📄 Examples with PMC IDs:")
                for paper in papers[:3]:
                    if paper.pmcid:
                        print(f"   PMID {paper.clean_pmid} -> {paper.pmcid}")
                        print(f"   Title: {paper.title[:80]}...")

        except Exception as e:
            print(f"❌ Error with query '{query}': {e}")

    print(f"\n🏁 Test complete!")

async def main():
    """Main test function"""
    await test_with_known_pmc_papers()

if __name__ == "__main__":
    asyncio.run(main())