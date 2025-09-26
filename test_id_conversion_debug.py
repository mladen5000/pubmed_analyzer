#!/usr/bin/env python3
"""
Debug script to test PMID to PMC ID conversion specifically
"""

import asyncio
import aiohttp
import json
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Test PMIDs from the recent results
TEST_PMIDS = ["41004260", "41004259", "41004253", "41004217", "41004210"]

async def test_bulk_conversion(pmids, email="mrasic2@uic.edu"):
    """Test the bulk ID conversion API directly"""

    try:
        # Join PMIDs for bulk query
        pmid_string = ",".join(pmids)

        print(f"Testing bulk conversion for PMIDs: {pmid_string}")

        # Use the ID Converter API endpoint
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
            print(f"Response headers: {dict(response.headers)}")

            if response.status == 200:
                data = await response.json()
                print(f"Raw response data: {json.dumps(data, indent=2)}")

                # Parse the conversion results
                pmid_to_pmcid = {}

                if 'records' in data:
                    for record in data['records']:
                        pmid = record.get('pmid')
                        pmcid = record.get('pmcid')
                        status = record.get('status', 'unknown')

                        print(f"Record: PMID={pmid}, PMC={pmcid}, Status={status}")

                        if pmid and pmcid:
                            # Convert PMID to string for consistent key matching
                            pmid_str = str(pmid)

                            # Clean up PMC ID format
                            if pmcid.startswith('PMC'):
                                pmid_to_pmcid[pmid_str] = pmcid
                            else:
                                pmid_to_pmcid[pmid_str] = f'PMC{pmcid}'

                print(f"Converted {len(pmid_to_pmcid)}/{len(pmids)} PMIDs to PMC IDs")
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

async def test_individual_conversion(pmid, email="mrasic2@uic.edu"):
    """Test individual ID conversion via eLink"""

    try:
        print(f"\nTesting individual conversion for PMID: {pmid}")

        # Test eLink conversion
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
        params = {
            'dbfrom': 'pubmed',
            'db': 'pmc',
            'id': pmid,
            'retmode': 'json',
            'tool': 'PubMedAnalyzer',
            'email': email
        }

        async with aiohttp.ClientSession() as session:
            response = await session.get(url, params=params)

            print(f"eLink response status: {response.status}")

            if response.status == 200:
                data = await response.json()
                print(f"eLink raw response: {json.dumps(data, indent=2)}")

                # Extract PMC ID from elink response
                linksets = data.get('linksets', [])
                for linkset in linksets:
                    linksetdbs = linkset.get('linksetdbs', [])
                    for linksetdb in linksetdbs:
                        if linksetdb.get('dbto') == 'pmc':
                            links = linksetdb.get('links', [])
                            if links:
                                pmcid = f"PMC{links[0]}"
                                print(f"Found PMC ID via eLink: {pmcid}")
                                return pmcid

                print("No PMC ID found via eLink")
                return None

            else:
                response_text = await response.text()
                print(f"eLink returned status {response.status}")
                print(f"Response body: {response_text}")
                return None

    except Exception as e:
        print(f"Error in individual conversion: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_pmc_search(pmid, email="mrasic2@uic.edu"):
    """Test PMC search for PMID"""

    try:
        print(f"\nTesting PMC search for PMID: {pmid}")

        # Test PMC search
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            'db': 'pmc',
            'term': f'{pmid}[PMID]',
            'retmax': '1',
            'retmode': 'json',
            'tool': 'PubMedAnalyzer',
            'email': email
        }

        async with aiohttp.ClientSession() as session:
            response = await session.get(url, params=params)

            print(f"PMC search response status: {response.status}")

            if response.status == 200:
                data = await response.json()
                print(f"PMC search raw response: {json.dumps(data, indent=2)}")

                search_result = data.get('esearchresult', {})
                ids = search_result.get('idlist', [])

                if ids:
                    pmcid = f"PMC{ids[0]}"
                    print(f"Found PMC ID via search: {pmcid}")
                    return pmcid
                else:
                    print("No PMC ID found via search")
                    return None

            else:
                response_text = await response.text()
                print(f"PMC search returned status {response.status}")
                print(f"Response body: {response_text}")
                return None

    except Exception as e:
        print(f"Error in PMC search: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    print("=" * 60)
    print("PMID to PMC ID Conversion Debug Test")
    print("=" * 60)

    # Test bulk conversion first
    print("\n1. Testing bulk ID converter API...")
    bulk_results = await test_bulk_conversion(TEST_PMIDS)

    # Test individual methods for the first PMID
    test_pmid = TEST_PMIDS[0]

    print(f"\n2. Testing individual methods for PMID {test_pmid}...")
    elink_result = await test_individual_conversion(test_pmid)
    search_result = await test_pmc_search(test_pmid)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Bulk conversion results: {len(bulk_results)} successful conversions")
    print(f"Individual eLink result for {test_pmid}: {elink_result}")
    print(f"Individual search result for {test_pmid}: {search_result}")

    if not bulk_results and not elink_result and not search_result:
        print("\n⚠️  WARNING: All conversion methods failed!")
        print("This suggests these PMIDs may not have corresponding PMC IDs.")
        print("This is normal for very recent papers or papers not in PMC.")

if __name__ == "__main__":
    asyncio.run(main())