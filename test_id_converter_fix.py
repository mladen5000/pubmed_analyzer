#!/usr/bin/env python3
"""
Test script to diagnose and fix the PMC ID conversion issue
"""

import asyncio
import json
import logging
import os
import sys
from typing import List

# Add the pubmed_analyzer directory to the path
sys.path.insert(0, '/Users/mladenrasic/Projects/pubmed_analyzer')

from pubmed_analyzer.core.id_converter import PMIDToPMCConverter
from pubmed_analyzer.models.paper import Paper

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_current_implementation():
    """Test the current ID converter implementation"""

    # Test with some known PMIDs that should have PMC IDs
    test_pmids = [
        "36308294",  # Recent paper that should have PMC ID
        "35395086",  # Another recent paper
        "34662866",  # Another test
        "33686204",  # COVID-19 paper (likely to have PMC ID)
        "32887691"   # Another COVID paper
    ]

    print("🔬 Testing current ID converter implementation...")
    print(f"Test PMIDs: {test_pmids}")

    # Create test papers
    papers = []
    for i, pmid in enumerate(test_pmids):
        paper = Paper(
            pmid=f"PMID:{pmid}",
            title=f"Test Paper {i+1}",
            authors=["Test Author"],
            abstract="Test abstract",
            journal="Test Journal"
        )
        papers.append(paper)

    # Get email from environment or use default
    email = os.getenv('PUBMED_EMAIL', 'test@example.com')
    api_key = os.getenv('NCBI_API_KEY')

    print(f"Using email: {email}")
    print(f"Using API key: {'Yes' if api_key else 'No'}")

    # Test the current converter
    converter = PMIDToPMCConverter(email=email, api_key=api_key)

    try:
        await converter.enrich_with_pmcids(papers, batch_size=5)

        print("\n📊 Results from current implementation:")
        successful_conversions = 0
        for paper in papers:
            pmcid_status = paper.pmcid if paper.pmcid else "❌ No PMC ID"
            if paper.pmcid:
                successful_conversions += 1
            print(f"  PMID {paper.clean_pmid} -> {pmcid_status}")

        print(f"\n✅ Success rate: {successful_conversions}/{len(papers)} ({successful_conversions/len(papers)*100:.1f}%)")

        if successful_conversions == 0:
            print("❌ ZERO conversions - this confirms the bug!")
            return False
        else:
            print("✅ Some conversions succeeded - implementation may be working")
            return True

    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_manual_api_call():
    """Test the API manually to see what's wrong"""

    print("\n🔧 Testing manual API call to identify the issue...")

    import aiohttp

    # Test with the CURRENT (wrong) endpoint
    wrong_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

    # Test with the CORRECT endpoint from official docs
    correct_url = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"

    test_pmids = ["36308294", "35395086"]
    email = os.getenv('PUBMED_EMAIL', 'test@example.com')

    async with aiohttp.ClientSession() as session:

        # Test WRONG endpoint
        print(f"\n🔴 Testing WRONG endpoint: {wrong_url}")
        try:
            params = {
                'tool': 'PubMedAnalyzer',
                'email': email,
                'ids': ','.join(test_pmids),
                'format': 'json',
                'versions': 'no'
            }

            async with session.get(wrong_url, params=params) as response:
                print(f"Status: {response.status}")
                text = await response.text()
                print(f"Response: {text[:500]}...")

        except Exception as e:
            print(f"Error with wrong endpoint: {e}")

        # Test CORRECT endpoint
        print(f"\n✅ Testing CORRECT endpoint: {correct_url}")
        try:
            # Use the correct parameters from official docs
            params = {
                'tool': 'PubMedAnalyzer',
                'email': email,
                'ids': ','.join(test_pmids),
                'format': 'json'
            }

            async with session.get(correct_url, params=params) as response:
                print(f"Status: {response.status}")
                if response.status == 200:
                    try:
                        data = await response.json()
                        print(f"JSON Response: {json.dumps(data, indent=2)}")
                    except:
                        text = await response.text()
                        print(f"Text Response: {text[:500]}...")
                else:
                    text = await response.text()
                    print(f"Error Response: {text[:500]}...")

        except Exception as e:
            print(f"Error with correct endpoint: {e}")

async def main():
    """Main test function"""
    print("🚀 Starting PMC ID Converter Diagnostic Test")
    print("=" * 60)

    # Test current implementation
    current_works = await test_current_implementation()

    # Test manual API calls
    await test_manual_api_call()

    print("\n" + "=" * 60)
    if not current_works:
        print("🔧 DIAGNOSIS: Current implementation is broken - needs fixing!")
        print("💡 SOLUTION: Update to use correct API endpoint and parameters")
    else:
        print("✅ Current implementation seems to work")

if __name__ == "__main__":
    asyncio.run(main())