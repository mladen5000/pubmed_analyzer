#!/usr/bin/env python3
import os
import csv
import logging
from Bio import Entrez
import boto3
from botocore import UNSIGNED
from botocore.client import Config

from boto3.session import Config as BotoConfig
from xml.etree import ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# ----------------------------
# CONFIGURATION
# ----------------------------
Entrez.email = "mrasic2@uic.edu"
S3_BUCKET = "pmc-oa-opendata"
OA_GROUPS = ["oa_comm", "oa_noncomm", "phe_timebound"]
METADATA_CSV_PREFIX = "xml/metadata/csv"
TXT_PREFIX = "txt/all"
XML_PREFIX = "xml/all"
DOWNLOAD_DIR = "./downloads"
MAX_RESULTS = 50
MAX_WORKERS = 10
RETRIES = 3

# ----------------------------
# LOGGING
# ----------------------------
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ----------------------------
# S3 CLIENT (unsigned)
# ----------------------------
# s3_client = boto3.client("s3", config=BotoConfig(signature_version="unsigned"))
# S3 client for public OA bucket
s3_client = boto3.client(
    "s3",
    config=Config(signature_version=UNSIGNED)
)


# ----------------------------
# FETCH CSV METADATA
# ----------------------------
def fetch_metadata_csv(group):
    csv_key = f"{group}/{METADATA_CSV_PREFIX}/{group}.filelist.csv"
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=csv_key)
        content = obj["Body"].read().decode("utf-8").splitlines()
        reader = csv.DictReader(content)
        records = [row for row in reader]
        logger.info(f"Fetched {len(records)} records from {group}")
        return records
    except Exception as e:
        logger.error(f"Failed to fetch metadata CSV for {group}: {e}")
        return []


def fetch_all_metadata():
    pmc_map = {}
    for group in OA_GROUPS:
        records = fetch_metadata_csv(group)
        for rec in records:
            pmcid = rec.get("AccessionID")
            if pmcid:
                rec["_oa_group"] = group
                pmc_map[pmcid] = rec
    logger.info(f"Total PMCIDs in OA subset: {len(pmc_map)}")
    return pmc_map


# ----------------------------
# PUBMED SEARCH & METADATA
# ----------------------------
def search_pubmed(keywords):
    """Search PubMed for keywords and only return articles with PMCIDs."""
    query = f"{keywords} AND pmc[filter]"
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=MAX_RESULTS)
        record = Entrez.read(handle)
        handle.close()
        pmids = record.get("IdList", [])
        logger.info(f"Found {len(pmids)} PubMed IDs with PMCIDs")
        return pmids
    except Exception as e:
        logger.error(f"PubMed search failed: {e}")
        return []


def fetch_pubmed_metadata(pmids):
    if not pmids:
        return []
    try:
        handle = Entrez.efetch(db="pubmed", id=pmids, rettype="xml", retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        return records.get("PubmedArticle", [])
    except Exception as e:
        logger.error(f"Fetching PubMed metadata failed: {e}")
        return []


def extract_pmcid_from_article(article):
    try:
        for aid in article.get("PubmedData", {}).get("ArticleIdList", []):
            if aid.attributes.get("IdType") == "pmc":
                return aid
    except Exception:
        pass
    return None


# ----------------------------
# DOWNLOAD & PARSE
# ----------------------------
def fetch_from_s3(group, pmc_id, extension, download_dir):
    key = f"{group}/{extension}/{pmc_id}.{'txt' if extension == 'txt/all' else 'xml'}"
    local_path = os.path.join(
        download_dir, f"{pmc_id}.{'txt' if extension == 'txt/all' else 'xml'}"
    )
    attempt = 0
    while attempt < RETRIES:
        try:
            os.makedirs(download_dir, exist_ok=True)
            s3_client.download_file(S3_BUCKET, key, local_path)
            return local_path
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed to download {key}: {e}")
            attempt += 1
            time.sleep(1)
    return None


def download_article(pmc_id, metadata_map, download_dir):
    rec = metadata_map.get(pmc_id)
    if not rec:
        logger.info(f"PMC {pmc_id} not in OA CSV, skipping.")
        return None
    group = rec["_oa_group"]
    path = fetch_from_s3(group, pmc_id, TXT_PREFIX, download_dir)
    if path:
        return path
    path = fetch_from_s3(group, pmc_id, XML_PREFIX, download_dir)
    return path


def parse_abstract_from_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        abstract_elem = root.find(".//abstract/p")
        if abstract_elem is not None and abstract_elem.text:
            return abstract_elem.text.strip()
    except Exception as e:
        logger.warning(f"Failed to parse abstract from {xml_path}: {e}")
    return None


# ----------------------------
# PARALLEL WORKER
# ----------------------------
def process_article(pmc_id, metadata_map, download_dir):
    try:
        local_file = download_article(pmc_id, metadata_map, download_dir)
        abstract = None
        if local_file and local_file.endswith(".xml"):
            abstract = parse_abstract_from_xml(local_file)
        return (pmc_id, local_file, abstract)
    except Exception as e:
        logger.error(f"Error processing PMC {pmc_id}: {e}")
        return (pmc_id, None, None)


# ----------------------------
# MAIN WORKFLOW
# ----------------------------
def main(keywords, download_dir=DOWNLOAD_DIR):
    metadata_map = fetch_all_metadata()
    pmids = search_pubmed(keywords)
    articles = fetch_pubmed_metadata(pmids)

    pmc_ids = []
    for article in articles:
        pmc_id = extract_pmcid_from_article(article)
        if pmc_id:
            pmc_ids.append(pmc_id)

    if not pmc_ids:
        logger.info("No PMCIDs found in PubMed search with OA filter.")
        return

    logger.info(f"Processing {len(pmc_ids)} PMC articles in parallel...")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_pmc = {
            executor.submit(process_article, pmc_id, metadata_map, download_dir): pmc_id
            for pmc_id in pmc_ids
        }
        for future in as_completed(future_to_pmc):
            pmc_id, path, abstract = future.result()
            results.append((pmc_id, path, abstract))
            if path:
                logger.info(f"Downloaded {pmc_id}: {path}")
            if abstract:
                logger.info(f"Abstract ({pmc_id}): {abstract[:200]}...")


if __name__ == "__main__":
    keywords = "metagenomics bioinformatics"
    main(keywords)
