#!/usr/bin/env python3
"""
Fixed PMC OA Service Strategy for PDF Downloads
This demonstrates the correct way to use the PMC OA Service API
"""

import asyncio
import aiohttp
import xml.etree.ElementTree as ET
import logging
from typing import Optional, Dict, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class FixedPMCOAServiceStrategy:
    """Correctly implemented PMC Open Access service strategy"""

    @property
    def name(self) -> str:
        return "Fixed PMC OA Service"

    async def get_pdf_info(self, session: aiohttp.ClientSession, pmc_id: str) -> Optional[Dict]:
        """
        Get PDF information from PMC OA Service using correct API

        Args:
            session: HTTP session
            pmc_id: PMC ID (with or without PMC prefix)

        Returns:
            Dictionary with PDF URL and metadata, or None if not available
        """
        try:
            # Clean PMC ID
            clean_pmc = pmc_id.replace('PMC', '')
            full_pmc = f"PMC{clean_pmc}"

            # Use the correct PMC OA Service endpoint
            url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
            params = {'id': full_pmc}

            headers = {
                'User-Agent': 'PubMedAnalyzer/1.0 (https://example.com/contact)',
                'Accept': 'application/xml, text/xml',
            }

            async with session.get(url, params=params, headers=headers) as response:
                if response.status != 200:
                    logger.debug(f"PMC OA Service returned {response.status} for {full_pmc}")
                    return None

                xml_content = await response.text()

                # Parse XML response
                try:
                    root = ET.fromstring(xml_content)

                    # Check for errors
                    error = root.find('error')
                    if error is not None:
                        logger.debug(f"PMC OA Service error for {full_pmc}: {error.text}")
                        return None

                    # Find PDF links
                    records = root.find('records')
                    if records is None:
                        return None

                    record = records.find('record')
                    if record is None:
                        return None

                    # Extract metadata
                    metadata = {
                        'pmcid': full_pmc,
                        'citation': record.get('citation', 'N/A'),
                        'license': record.get('license', 'N/A'),
                        'retracted': record.get('retracted', 'no') == 'yes'
                    }

                    # Find PDF link
                    pdf_link = None
                    for link in record.findall('link'):
                        link_format = link.get('format', '').lower()
                        if 'pdf' in link_format:
                            href = link.get('href')
                            if href:
                                # Convert FTP to HTTPS
                                if href.startswith('ftp://ftp.ncbi.nlm.nih.gov'):
                                    href = href.replace('ftp://ftp.ncbi.nlm.nih.gov', 'https://ftp.ncbi.nlm.nih.gov')
                                pdf_link = href
                                break

                    if pdf_link:
                        metadata['pdf_url'] = pdf_link
                        logger.debug(f"Found PDF for {full_pmc}: {pdf_link}")
                        return metadata
                    else:
                        logger.debug(f"No PDF link found for {full_pmc}")
                        return None

                except ET.ParseError as e:
                    logger.error(f"Failed to parse XML response for {full_pmc}: {e}")
                    return None

        except Exception as e:
            logger.error(f"PMC OA Service request failed for {pmc_id}: {e}")
            return None


async def test_fixed_strategy():
    """Test the fixed strategy with known good PMC IDs"""

    # Known open access PMC IDs with PDFs
    test_pmcids = [
        "PMC6557568",  # PLoS One article
        "PMC7308628",  # Nature article
        "PMC8443998",  # Another PLoS One
        "PMC9123456",  # Test non-existent
    ]

    strategy = FixedPMCOAServiceStrategy()

    async with aiohttp.ClientSession() as session:
        for pmc_id in test_pmcids:
            print(f"\nTesting {pmc_id}:")

            result = await strategy.get_pdf_info(session, pmc_id)

            if result:
                print(f"  ✅ Success!")
                print(f"  📄 Citation: {result['citation']}")
                print(f"  📜 License: {result['license']}")
                print(f"  📁 PDF URL: {result['pdf_url']}")

                # Test if PDF URL actually works
                try:
                    async with session.head(result['pdf_url'], timeout=aiohttp.ClientTimeout(total=10)) as pdf_response:
                        print(f"  🔍 PDF Status: {pdf_response.status}")
                        print(f"  📏 Content-Type: {pdf_response.headers.get('content-type', 'Unknown')}")
                except Exception as e:
                    print(f"  ❌ PDF URL test failed: {e}")
            else:
                print(f"  ❌ No PDF found")


if __name__ == "__main__":
    asyncio.run(test_fixed_strategy())