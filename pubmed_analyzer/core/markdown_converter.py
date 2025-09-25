import os
import logging
from pathlib import Path
from typing import List, Optional, Dict
from tqdm import tqdm

try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False
    logging.warning("MarkItDown not available - PDF to Markdown conversion will be skipped")

from ..models.paper import Paper

logger = logging.getLogger(__name__)


class MarkdownConverter:
    """Convert PDFs to Markdown format using MarkItDown"""

    def __init__(self, markdown_dir: str = "markdown"):
        self.markdown_dir = markdown_dir

        if not MARKITDOWN_AVAILABLE:
            logger.error("MarkItDown library is not installed. Install with: pip install markitdown")
            return

        # Initialize MarkItDown with minimal plugins for better performance
        self.md_converter = MarkItDown(enable_plugins=False)

        # Create markdown directory
        os.makedirs(self.markdown_dir, exist_ok=True)
        logger.info(f"Markdown converter initialized - output directory: {self.markdown_dir}")

    def is_available(self) -> bool:
        """Check if MarkItDown is available"""
        return MARKITDOWN_AVAILABLE

    def convert_single_pdf(self, pdf_path: str, output_name: Optional[str] = None) -> Optional[str]:
        """
        Convert a single PDF to Markdown

        Args:
            pdf_path: Path to the PDF file
            output_name: Optional custom output filename (without extension)

        Returns:
            Path to the created Markdown file, or None if conversion failed
        """
        if not self.is_available():
            logger.error("MarkItDown not available")
            return None

        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return None

        try:
            # Generate output filename
            if output_name:
                md_filename = f"{output_name}.md"
            else:
                pdf_name = Path(pdf_path).stem
                md_filename = f"{pdf_name}.md"

            md_path = os.path.join(self.markdown_dir, md_filename)

            logger.debug(f"Converting {pdf_path} -> {md_path}")

            # Convert PDF to markdown
            result = self.md_converter.convert(pdf_path)

            # Save markdown content
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(result.text_content)

            # Validate the conversion
            if os.path.exists(md_path) and os.path.getsize(md_path) > 100:  # At least 100 bytes
                logger.debug(f"Successfully converted {pdf_path} -> {md_path}")
                return md_path
            else:
                logger.warning(f"Conversion resulted in empty or too small file: {md_path}")
                return None

        except Exception as e:
            logger.error(f"Failed to convert {pdf_path} to Markdown: {e}")
            return None

    def convert_papers_pdfs(self, papers: List[Paper]) -> Dict[str, str]:
        """
        Convert PDFs from a list of Paper objects to Markdown

        Args:
            papers: List of Paper objects with pdf_path set

        Returns:
            Dictionary mapping PMC ID to markdown file path
        """
        if not self.is_available():
            logger.error("MarkItDown not available")
            return {}

        # Filter papers with valid PDF paths
        papers_with_pdfs = [p for p in papers if p.pdf_path and os.path.exists(p.pdf_path)]

        if not papers_with_pdfs:
            logger.warning("No papers with valid PDF files found")
            return {}

        logger.info(f"Converting {len(papers_with_pdfs)} PDFs to Markdown format")

        results = {}
        successful_conversions = 0

        # Convert each PDF with progress tracking
        for paper in tqdm(papers_with_pdfs, desc="Converting PDFs to Markdown", unit="pdf"):
            try:
                # Use clean PMC ID as filename
                output_name = paper.clean_pmcid if paper.pmcid else f"pmid_{paper.clean_pmid}"
                md_path = self.convert_single_pdf(paper.pdf_path, output_name)

                if md_path:
                    results[paper.pmcid or paper.pmid] = md_path
                    paper.markdown_path = md_path  # Store markdown path in paper object
                    successful_conversions += 1

            except Exception as e:
                logger.error(f"Failed to convert PDF for {paper.pmcid or paper.pmid}: {e}")

        conversion_rate = (successful_conversions / len(papers_with_pdfs)) * 100
        logger.info(f"Markdown conversion results: {successful_conversions}/{len(papers_with_pdfs)} "
                   f"successful ({conversion_rate:.1f}%)")

        return results

    def convert_pdf_directory(self, pdf_dir: str = "pdfs") -> Dict[str, str]:
        """
        Convert all PDFs in a directory to Markdown

        Args:
            pdf_dir: Directory containing PDF files

        Returns:
            Dictionary mapping PDF filename to markdown file path
        """
        if not self.is_available():
            logger.error("MarkItDown not available")
            return {}

        if not os.path.exists(pdf_dir):
            logger.error(f"PDF directory not found: {pdf_dir}")
            return {}

        # Find all PDF files
        pdf_files = []
        for filename in os.listdir(pdf_dir):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(pdf_dir, filename)
                if os.path.isfile(pdf_path):
                    pdf_files.append(pdf_path)

        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return {}

        logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

        results = {}
        successful_conversions = 0

        # Convert each PDF with progress tracking
        for pdf_path in tqdm(pdf_files, desc="Converting PDFs to Markdown", unit="pdf"):
            try:
                filename = Path(pdf_path).stem
                md_path = self.convert_single_pdf(pdf_path, filename)

                if md_path:
                    results[filename] = md_path
                    successful_conversions += 1

            except Exception as e:
                logger.error(f"Failed to convert {pdf_path}: {e}")

        conversion_rate = (successful_conversions / len(pdf_files)) * 100
        logger.info(f"Directory conversion results: {successful_conversions}/{len(pdf_files)} "
                   f"successful ({conversion_rate:.1f}%)")

        return results

    def get_conversion_stats(self) -> Dict[str, int]:
        """Get statistics about converted markdown files"""
        if not os.path.exists(self.markdown_dir):
            return {'total_files': 0, 'total_size_mb': 0}

        md_files = [f for f in os.listdir(self.markdown_dir) if f.endswith('.md')]
        total_size = sum(os.path.getsize(os.path.join(self.markdown_dir, f)) for f in md_files)

        return {
            'total_files': len(md_files),
            'total_size_mb': round(total_size / (1024 * 1024), 2)
        }