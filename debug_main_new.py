#!/usr/bin/env python3
"""
Debug script to test main_new.py PMC ID conversion in detail
"""

import asyncio
import json
import logging
import os
import sys

# Add the pubmed_analyzer directory to the path
sys.path.insert(0, '/Users/mladenrasic/Projects/pubmed_analyzer')

from pubmed_analyzer.core.search import PubMedSearcher
from pubmed_analyzer.core.id_converter import PMIDToPMCConverter

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def debug_main_new_workflow():
    """Debug the exact workflow that main_new.py uses"""

    print("🔬 Debugging main_new.py PMC ID conversion workflow...")

    # Use the same configuration as main_new.py
    EMAIL = "mrasic2@uic.edu"
    NCBI_API_KEY = os.getenv("NCBI_API_KEY")

    print(f"Using email: {EMAIL}")
    print(f"Using API key: {'Yes' if NCBI_API_KEY else 'No'}")

    # Step 1: Search exactly like main_new.py
    searcher = PubMedSearcher(EMAIL, NCBI_API_KEY)

    print("\n📡 Step 1: Searching PubMed...")
    pmids = await searcher.search_papers(
        query="COVID-19",
        max_results=5
    )

    print(f"Found PMIDs: {pmids}")

    # Step 2: Fetch metadata exactly like main_new.py
    print("\n📄 Step 2: Fetching metadata...")
    papers = await searcher.fetch_papers_metadata(pmids)

    print(f"\n📊 Papers retrieved:")
    for i, paper in enumerate(papers):
        print(f"  {i+1}. PMID: {paper.pmid} (clean: {paper.clean_pmid})")
        print(f"      Title: {paper.title[:100]}...")
        print(f"      PMC ID: {paper.pmcid}")
        print(f"      Has fulltext: {paper.has_fulltext}")
        print(f"      Abstract length: {len(paper.abstract) if paper.abstract else 0}")

    # Step 3: PMC ID conversion exactly like main_new.py
    print(f"\n🔄 Step 3: PMC ID conversion...")
    id_converter = PMIDToPMCConverter(EMAIL, NCBI_API_KEY)

    # Test the bulk conversion directly
    print("\n🧪 Testing bulk conversion function directly...")
    clean_pmids = [paper.clean_pmid for paper in papers if paper.clean_pmid]
    print(f"Clean PMIDs to convert: {clean_pmids}")

    # Create a test client to see what the API returns
    from pubmed_analyzer.utils.ncbi_client import NCBIClient

    async with NCBIClient(EMAIL, NCBI_API_KEY) as client:
        pmid_to_pmcid = await id_converter._bulk_pmid_to_pmcid_conversion(client, clean_pmids)
        print(f"Bulk conversion result: {pmid_to_pmcid}")

    # Now run the actual enrichment
    print(f"\n🔄 Running actual enrichment...")
    await id_converter.enrich_with_pmcids(papers)

    print(f"\n📈 Final results:")
    pmcid_count = 0
    for i, paper in enumerate(papers):
        pmcid_status = paper.pmcid if paper.pmcid else "❌ No PMC ID"
        if paper.pmcid:
            pmcid_count += 1
        print(f"  {i+1}. PMID {paper.clean_pmid} -> {pmcid_status}")
        print(f"      Has fulltext: {paper.has_fulltext}")

    print(f"\n✅ Final success rate: {pmcid_count}/{len(papers)} ({pmcid_count/len(papers)*100:.1f}%)")

    # Let's also test the API call manually with these exact PMIDs
    print(f"\n🔧 Manual API test with actual PMIDs...")

    import aiohttp

    correct_url = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"

    async with aiohttp.ClientSession() as session:
        params = {
            'tool': 'PubMedAnalyzer',
            'email': EMAIL,
            'ids': ','.join(clean_pmids),
            'format': 'json'
        }

        async with session.get(correct_url, params=params) as response:
            print(f"Status: {response.status}")
            if response.status == 200:
                try:
                    data = await response.json()
                    print(f"API Response: {json.dumps(data, indent=2)}")
                except:
                    text = await response.text()
                    print(f"Text Response: {text[:1000]}...")
            else:
                text = await response.text()
                print(f"Error Response: {text[:500]}...")

async def main():
    """Main debug function"""
    await debug_main_new_workflow()

if __name__ == "__main__":
    asyncio.run(main())