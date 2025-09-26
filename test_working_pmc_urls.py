#!/usr/bin/env python3
"""
Test alternative working PMC URL patterns for PDF downloads
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


async def test_pmc_url_patterns(session: aiohttp.ClientSession, pmc_id: str) -> Dict[str, Optional[int]]:
    """Test various PMC URL patterns to see which ones work"""

    clean_pmc = pmc_id.replace('PMC', '')

    # Various URL patterns to test
    url_patterns = {
        'PMC Article PDF': f'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{clean_pmc}/pdf/',
        'PMC Article PDF (alt)': f'https://pmc.ncbi.nlm.nih.gov/articles/PMC{clean_pmc}/pdf/',
        'PMC Direct PDF': f'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{clean_pmc}/pdf/PMC{clean_pmc}.pdf',
        'PMC Utils PDF': f'https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC{clean_pmc}&format=pdf',
        'EuropePMC PDF': f'https://europepmc.org/articles/PMC{clean_pmc}?pdf=render',
        'EuropePMC Direct': f'https://europepmc.org/articles/pmc/PMC{clean_pmc}/pdf',
        'PMC FTP Style': f'https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/',  # Would need full path
    }

    results = {}

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'application/pdf,text/html,*/*',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    for name, url in url_patterns.items():
        if url.endswith('oa_pdf/'):  # Skip incomplete URL
            results[name] = None
            continue

        try:
            async with session.head(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15), allow_redirects=True) as response:
                results[name] = response.status
                if response.status == 200:
                    content_type = response.headers.get('content-type', '')
                    print(f"✅ {name}: {response.status} ({content_type})")
                    print(f"   URL: {url}")
                    if response.url != url:
                        print(f"   Final URL: {response.url}")
                elif response.status == 403:
                    print(f"🚫 {name}: {response.status} (Forbidden)")
                elif response.status in [301, 302, 307, 308]:
                    print(f"🔄 {name}: {response.status} (Redirect)")
                    location = response.headers.get('location', 'Unknown')
                    print(f"   Redirects to: {location}")
                else:
                    print(f"❌ {name}: {response.status}")
        except asyncio.TimeoutError:
            results[name] = 'TIMEOUT'
            print(f"⏰ {name}: Timeout")
        except Exception as e:
            results[name] = f'ERROR: {str(e)[:50]}'
            print(f"💥 {name}: {type(e).__name__}: {str(e)[:50]}")

    return results


async def test_specific_pmids_for_pdf_access():
    """Test specific PMIDs that are known to have open access PDFs"""

    # Test PMC IDs that should have PDFs
    test_cases = [
        "PMC8443998",  # Known to work from previous test
        "PMC6557568",  # PLoS One article
        "PMC7308628",  # Should be open access
        "PMC10000000", # Non-existent for comparison
    ]

    async with aiohttp.ClientSession() as session:
        for pmc_id in test_cases:
            print(f"\n{'='*60}")
            print(f"Testing PMC ID: {pmc_id}")
            print('='*60)

            results = await test_pmc_url_patterns(session, pmc_id)

            # Count successes
            success_count = sum(1 for status in results.values() if status == 200)
            print(f"\n📊 Summary for {pmc_id}: {success_count}/{len([r for r in results.values() if r is not None])} patterns worked")


async def test_doi_resolution():
    """Test DOI resolution patterns"""
    print(f"\n{'='*60}")
    print("Testing DOI Resolution Patterns")
    print('='*60)

    # Test DOI that should resolve to PDF
    test_doi = "10.1371/journal.pone.0218004"  # PLoS One article

    doi_patterns = {
        'Direct DOI': f'https://doi.org/{test_doi}',
        'DOI with PDF accept': f'https://doi.org/{test_doi}',  # with Accept: application/pdf header
        'Crossref API': f'https://api.crossref.org/works/{test_doi}',
    }

    async with aiohttp.ClientSession() as session:
        for name, url in doi_patterns.items():
            try:
                headers = {
                    'User-Agent': 'PubMedAnalyzer/1.0 (mailto:researcher@example.com)',
                    'Accept': 'application/pdf' if 'PDF accept' in name else 'text/html,application/json,*/*',
                }

                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15), allow_redirects=True) as response:
                    content_type = response.headers.get('content-type', '')
                    print(f"📄 {name}: {response.status} ({content_type})")
                    if response.url != url:
                        print(f"   Final URL: {response.url}")

                    if 'application/pdf' in content_type:
                        print(f"   🎉 This DOI resolves to a PDF!")

            except Exception as e:
                print(f"💥 {name}: {type(e).__name__}: {str(e)[:50]}")


if __name__ == "__main__":
    asyncio.run(test_specific_pmids_for_pdf_access())
    asyncio.run(test_doi_resolution())