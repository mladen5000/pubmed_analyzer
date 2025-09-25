#!/usr/bin/env python3
"""
Standalone script to convert existing PDFs to Markdown format using MarkItDown.
This script processes all PDF files in the pdfs/ directory and converts them to Markdown.
"""

import os
import sys
import logging
from pathlib import Path

# Add the pubmed_analyzer package to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pubmed_analyzer.core.markdown_converter import MarkdownConverter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Convert all existing PDFs to Markdown format"""

    # Check if pdfs directory exists
    pdf_dir = "pdfs"
    if not os.path.exists(pdf_dir):
        logger.error(f"PDF directory '{pdf_dir}' not found. Please run the main analysis first to download PDFs.")
        return False

    # Check if there are any PDF files
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logger.warning(f"No PDF files found in '{pdf_dir}' directory.")
        return False

    logger.info(f"🔍 Found {len(pdf_files)} PDF files to convert")

    # Initialize markdown converter
    markdown_converter = MarkdownConverter(markdown_dir="markdown")

    if not markdown_converter.is_available():
        logger.error("❌ MarkItDown library is not available. Please install it with: pip install markitdown")
        return False

    logger.info("📝 Starting PDF to Markdown conversion...")

    # Convert all PDFs in the directory
    results = markdown_converter.convert_pdf_directory(pdf_dir)

    if results:
        logger.info("✅ Conversion completed successfully!")

        # Display statistics
        stats = markdown_converter.get_conversion_stats()
        logger.info(f"📊 Conversion Statistics:")
        logger.info(f"   - Total files converted: {stats['total_files']}")
        logger.info(f"   - Total size: {stats['total_size_mb']} MB")
        logger.info(f"   - Output directory: markdown/")

        # List some converted files
        markdown_files = list(results.values())[:5]  # Show first 5
        logger.info(f"📁 Sample converted files:")
        for md_file in markdown_files:
            logger.info(f"   - {os.path.basename(md_file)}")

        if len(results) > 5:
            logger.info(f"   - ... and {len(results) - 5} more files")

        return True
    else:
        logger.error("❌ No PDFs were successfully converted to Markdown")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)