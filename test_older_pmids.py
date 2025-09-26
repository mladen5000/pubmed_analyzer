#!/usr/bin/env python3
"""
Test ID conversion with older PMIDs that are more likely to have PMC IDs
"""

import asyncio
import aiohttp
import json

# Test with some well-known older PMIDs that should have PMC IDs
OLDER_TEST_PMIDS = [
    "33117850",  # COVID-19 paper from 2020 - should have PMC
    "32007145",  # Early COVID-19 paper from early 2020
    "31978945",  # Original COVID-19 paper from China
    "32109013",  # Should have PMC ID
    "32007143"   # Another early COVID paper
]

async def test_bulk_conversion_older(pmids, email="mrasic2@uic.edu"):
    """Test the bulk ID conversion API with older PMIDs"""

    try:
        pmid_string = ",".join(pmids)
        print(f"Testing bulk conversion for older PMIDs: {pmid_string}")

        url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
        params = {
            'tool': 'PubMedAnalyzer',
            'email': email,
            'ids': pmid_string,
            'format': 'json',
            'versions': 'no'
        }

        async with aiohttp.ClientSession() as session:
            response = await session.get(url, params=params)

            print(f"Response status: {response.status}")

            if response.status == 200:
                data = await response.json()
                print(f"Raw response data: {json.dumps(data, indent=2)}")

                pmid_to_pmcid = {}

                if 'records' in data:
                    for record in data['records']:
                        pmid = record.get('pmid')
                        pmcid = record.get('pmcid')
                        status = record.get('status', 'unknown')

                        print(f"Record: PMID={pmid}, PMC={pmcid}, Status={status}")

                        if pmid and pmcid:
                            pmid_str = str(pmid)
                            if pmcid.startswith('PMC'):
                                pmid_to_pmcid[pmid_str] = pmcid
                            else:
                                pmid_to_pmcid[pmid_str] = f'PMC{pmcid}'

                print(f"Successfully converted {len(pmid_to_pmcid)}/{len(pmids)} PMIDs to PMC IDs")
                print(f"Conversion results: {pmid_to_pmcid}")
                return pmid_to_pmcid

            else:
                response_text = await response.text()
                print(f"API returned status {response.status}")
                print(f"Response body: {response_text}")
                return {}

    except Exception as e:
        print(f"Error in bulk conversion: {e}")
        import traceback
        traceback.print_exc()
        return {}

async def main():
    print("=" * 60)
    print("Testing with Older PMIDs (more likely to have PMC IDs)")
    print("=" * 60)

    results = await test_bulk_conversion_older(OLDER_TEST_PMIDS)

    print(f"\nResult: Found {len(results)} PMC IDs out of {len(OLDER_TEST_PMIDS)} PMIDs tested")

    if results:
        print("✅ ID conversion is working correctly!")
        print("The issue is that recent papers don't have PMC IDs yet.")
    else:
        print("❌ Still no conversions - there may be a deeper issue.")

if __name__ == "__main__":
    asyncio.run(main())