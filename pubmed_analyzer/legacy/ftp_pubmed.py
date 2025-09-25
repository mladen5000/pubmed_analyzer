import asyncio
import aiohttp
import os
import xml.etree.ElementTree as ET
from markitdown import MarkItDown

BASE_URL = "http://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

# Initialize MarkItDown once
md_converter = MarkItDown(enable_plugins=False)

async def fetch_pmc_record(session, pmcid, retries=3):
    url = f"{BASE_URL}?id={pmcid}"
    attempt = 0
    while attempt < retries:
        print(f"[DEBUG] Fetching {pmcid}, attempt {attempt+1}: {url}")
        try:
            async with session.get(url) as resp:
                print(f"[DEBUG] HTTP status for {pmcid}: {resp.status}")
                if resp.status != 200:
                    attempt += 1
                    continue
                text = await resp.text()
                root = ET.fromstring(text)
                record = root.find('records/record')
                if record is None:
                    print(f"[DEBUG] No OA record found for {pmcid}.")
                    return pmcid, None, None
                # Metadata
                metadata = {
                    'pmcid': pmcid,
                    'citation': record.attrib.get('citation', 'N/A'),
                    'license': record.attrib.get('license', 'N/A'),
                    'retracted': record.attrib.get('retracted', 'N/A'),
                }
                # Extract PDF URL from <link> elements
                pdf_url = None
                for link in record.findall('link'):
                    if link.attrib.get('format') == 'pdf':
                        pdf_url = link.attrib.get('href')
                        break
                print(f"[DEBUG] PDF URL for {pmcid}: {pdf_url}")
                return pmcid, metadata, pdf_url
        except Exception as e:
            print(f"[ERROR] Exception fetching {pmcid}: {e}")
            attempt += 1
    print(f"[ERROR] Failed to fetch {pmcid} after {retries} attempts")
    return pmcid, None, None

async def download_file(session, url, filename):
    if url is None:
        print(f"[DEBUG] No URL to download for {filename}")
        return
    print(f"[DEBUG] Downloading {url} -> {filename}")
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                content = await resp.read()
                with open(filename, 'wb') as f:
                    f.write(content)
                print(f"[DEBUG] Saved {filename}")
                # Convert PDF to text using MarkItDown
                convert_pdf_to_text(filename)
            else:
                print(f"[ERROR] Failed to download {url}: HTTP {resp.status}")
    except Exception as e:
        print(f"[ERROR] Exception downloading {url}: {e}")

def convert_pdf_to_text(pdf_path):
    try:
        result = md_converter.convert(pdf_path)
        txt_path = pdf_path.replace(".pdf", ".txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(result.text_content)
        print(f"[POSTPROCESS] Converted {pdf_path} -> {txt_path}")
    except Exception as e:
        print(f"[ERROR] MarkItDown conversion failed for {pdf_path}: {e}")

async def fetch_and_download_pdfs(pmcids, save_dir="downloads"):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_pmc_record(session, pmcid) for pmcid in pmcids]
        results = await asyncio.gather(*tasks)

        download_tasks = []
        for pmcid, metadata, pdf_url in results:
            if metadata:
                print(f"\n[INFO] PMCID: {pmcid}, Citation: {metadata['citation']}, License: {metadata['license']}")
                if pdf_url:
                    pdf_file = os.path.join(save_dir, "pdf", f"{pmcid}.pdf")
                    download_tasks.append(download_file(session, pdf_url, pdf_file))
            else:
                print(f"[INFO] No metadata or PDF for {pmcid}")

        if download_tasks:
            await asyncio.gather(*download_tasks)

# Example usage
pmcids = ["PMC5334499", "PMC6809050"]  # Add OA PMCIDs
asyncio.run(fetch_and_download_pdfs(pmcids))
