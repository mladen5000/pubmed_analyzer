#!/usr/bin/env python3
"""
Enhanced Document Parser for Section-Aware Analysis
Integrates with existing PDF processing pipeline and adds sophisticated section extraction
"""

import logging
import os
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

# Document processing imports
try:
    import fitz  # PyMuPDF - already used in existing codebase
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# NLP imports
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..models.paper import Paper

logger = logging.getLogger(__name__)


@dataclass
class DocumentStructure:
    """Structured representation of document layout"""
    title: Optional[str] = None
    authors: List[str] = None
    abstract: Optional[str] = None
    sections: Dict[str, str] = None
    figures: List[Dict[str, Any]] = None
    tables: List[Dict[str, Any]] = None
    references: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.authors is None:
            self.authors = []
        if self.sections is None:
            self.sections = {}
        if self.figures is None:
            self.figures = []
        if self.tables is None:
            self.tables = []
        if self.references is None:
            self.references = []
        if self.metadata is None:
            self.metadata = {}


class SciPDFParser:
    """
    Scientific PDF parser with advanced section detection
    Integrates with existing PyMuPDF pipeline while adding section intelligence
    """

    def __init__(self):
        self.nlp = None
        self.section_classifier = None

        # Initialize NLP components
        self._initialize_nlp()

        # Section detection patterns
        self.section_patterns = {
            'abstract': [
                r'^\s*abstract\s*$',
                r'^\s*summary\s*$',
                r'^\s*overview\s*$'
            ],
            'introduction': [
                r'^\s*1\.?\s*introduction\s*$',
                r'^\s*introduction\s*$',
                r'^\s*background\s*$',
                r'^\s*1\.?\s*background\s*$'
            ],
            'methods': [
                r'^\s*2\.?\s*methods?\s*$',
                r'^\s*methods?\s*$',
                r'^\s*methodology\s*$',
                r'^\s*materials?\s+and\s+methods?\s*$',
                r'^\s*experimental\s+procedures?\s*$',
                r'^\s*experimental\s+methods?\s*$'
            ],
            'results': [
                r'^\s*3\.?\s*results?\s*$',
                r'^\s*results?\s*$',
                r'^\s*findings?\s*$',
                r'^\s*observations?\s*$',
                r'^\s*experimental\s+results?\s*$'
            ],
            'discussion': [
                r'^\s*4\.?\s*discussion\s*$',
                r'^\s*discussion\s*$',
                r'^\s*interpretation\s*$',
                r'^\s*analysis\s*$'
            ],
            'conclusion': [
                r'^\s*5\.?\s*conclusions?\s*$',
                r'^\s*conclusions?\s*$',
                r'^\s*concluding\s+remarks?\s*$',
                r'^\s*final\s+remarks?\s*$',
                r'^\s*summary\s+and\s+conclusions?\s*$'
            ],
            'references': [
                r'^\s*references?\s*$',
                r'^\s*bibliography\s*$',
                r'^\s*literature\s+cited\s*$'
            ],
            'acknowledgments': [
                r'^\s*acknowledgments?\s*$',
                r'^\s*acknowledgements?\s*$'
            ]
        }

        logger.info("🔬 Enhanced Scientific PDF Parser initialized")
        logger.info(f"   PyMuPDF: {'✅' if PYMUPDF_AVAILABLE else '❌'}")
        logger.info(f"   PDFPlumber: {'✅' if PDFPLUMBER_AVAILABLE else '❌'}")
        logger.info(f"   spaCy NLP: {'✅' if SPACY_AVAILABLE else '❌'}")

    def _initialize_nlp(self):
        """Initialize NLP components"""
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("   📚 Loaded spaCy model for advanced text processing")
            except OSError:
                logger.warning("   ⚠️  spaCy model not found - install with: python -m spacy download en_core_web_sm")

        if TRANSFORMERS_AVAILABLE:
            try:
                # Initialize section classifier (could be fine-tuned BERT model)
                self.section_classifier = pipeline(
                    "text-classification",
                    model="microsoft/DialoGPT-medium",  # Placeholder - would use custom section classifier
                    return_all_scores=True
                )
                logger.info("   🤖 Loaded transformer-based section classifier")
            except Exception as e:
                logger.warning(f"   ⚠️  Could not load transformer model: {e}")

    def parse_pdf_with_sections(self, pdf_path: str, paper: Paper = None) -> DocumentStructure:
        """
        Parse PDF with advanced section detection

        Args:
            pdf_path: Path to PDF file
            paper: Optional Paper object to enrich with parsed data

        Returns:
            DocumentStructure with extracted sections and metadata
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return DocumentStructure()

        logger.info(f"🔍 Parsing PDF with section detection: {Path(pdf_path).name}")

        try:
            # Primary parsing with PyMuPDF (existing pipeline)
            structure = self._parse_with_pymupdf(pdf_path)

            # Enhanced parsing with PDFPlumber if available
            if PDFPLUMBER_AVAILABLE:
                enhanced_structure = self._enhance_with_pdfplumber(pdf_path, structure)
                structure = self._merge_structures(structure, enhanced_structure)

            # Advanced section classification with NLP
            if self.nlp:
                structure = self._enhance_with_nlp(structure)

            # Update Paper object if provided
            if paper:
                self._update_paper_with_structure(paper, structure)

            logger.info(f"✅ Extracted {len(structure.sections)} sections from PDF")
            return structure

        except Exception as e:
            logger.error(f"❌ Failed to parse PDF {pdf_path}: {e}")
            return DocumentStructure()

    def _parse_with_pymupdf(self, pdf_path: str) -> DocumentStructure:
        """Parse PDF using PyMuPDF (integrates with existing pipeline)"""
        if not PYMUPDF_AVAILABLE:
            logger.error("PyMuPDF not available")
            return DocumentStructure()

        structure = DocumentStructure()

        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            page_texts = []

            # Extract text from all pages
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                page_texts.append(page_text)
                full_text += page_text + "\n"

            doc.close()

            # Extract basic structure
            structure.metadata = {
                "total_pages": len(page_texts),
                "parser": "pymupdf",
                "file_path": pdf_path
            }

            # Basic section detection using patterns
            sections = self._detect_sections_with_patterns(full_text)
            structure.sections = sections

            # Extract title and abstract from first pages
            first_page_text = page_texts[0] if page_texts else ""
            structure.title = self._extract_title(first_page_text)
            structure.abstract = self._extract_abstract(first_page_text, page_texts[1] if len(page_texts) > 1 else "")

            return structure

        except Exception as e:
            logger.error(f"PyMuPDF parsing failed: {e}")
            return DocumentStructure()

    def _enhance_with_pdfplumber(self, pdf_path: str, base_structure: DocumentStructure) -> DocumentStructure:
        """Enhance parsing with PDFPlumber for better structure detection"""
        enhanced_structure = DocumentStructure()

        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                # Extract text with better formatting preservation
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"

                # Extract tables
                tables = []
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for table_num, table in enumerate(page_tables):
                        tables.append({
                            "page": page_num + 1,
                            "table_number": table_num + 1,
                            "data": table,
                            "text_summary": self._summarize_table(table)
                        })

                enhanced_structure.tables = tables
                enhanced_structure.metadata = {
                    "parser": "pdfplumber",
                    "tables_found": len(tables)
                }

                # Re-run section detection with better formatted text
                if full_text:
                    enhanced_sections = self._detect_sections_with_patterns(full_text)
                    enhanced_structure.sections = enhanced_sections

            return enhanced_structure

        except Exception as e:
            logger.error(f"PDFPlumber enhancement failed: {e}")
            return DocumentStructure()

    def _detect_sections_with_patterns(self, text: str) -> Dict[str, str]:
        """Detect sections using regex patterns"""
        sections = {}
        text_lines = text.split('\n')

        # Find section boundaries
        section_boundaries = []

        for line_num, line in enumerate(text_lines):
            line_clean = line.strip().lower()

            for section_type, patterns in self.section_patterns.items():
                for pattern in patterns:
                    if re.match(pattern, line_clean, re.IGNORECASE):
                        section_boundaries.append((line_num, section_type, line.strip()))
                        break

        # Extract section contents
        for i, (start_line, section_type, header) in enumerate(section_boundaries):
            # Find end of section (next section or end of document)
            end_line = len(text_lines)
            if i + 1 < len(section_boundaries):
                end_line = section_boundaries[i + 1][0]

            # Extract section text
            section_lines = text_lines[start_line + 1:end_line]
            section_text = '\n'.join(section_lines).strip()

            # Only include sections with substantial content
            if len(section_text) > 100:
                sections[section_type] = section_text

        # If no clear sections found, try to split by common headers
        if not sections:
            sections = self._fallback_section_detection(text)

        return sections

    def _fallback_section_detection(self, text: str) -> Dict[str, str]:
        """Fallback section detection using simple heuristics"""
        sections = {}

        # Look for numbered sections
        numbered_pattern = r'\n\s*(\d+\.?\s+[A-Z][a-zA-Z\s]+)\n'
        matches = list(re.finditer(numbered_pattern, text))

        if matches:
            for i, match in enumerate(matches):
                start_pos = match.end()
                end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)

                section_title = match.group(1).strip()
                section_content = text[start_pos:end_pos].strip()

                if len(section_content) > 100:
                    # Classify section based on title
                    section_type = self._classify_section_by_title(section_title)
                    sections[section_type] = section_content

        # If still no sections, treat entire text as unknown section
        if not sections and len(text) > 500:
            sections['full_text'] = text

        return sections

    def _classify_section_by_title(self, title: str) -> str:
        """Classify section type based on title"""
        title_lower = title.lower()

        classification_keywords = {
            'introduction': ['introduction', 'background', 'overview'],
            'methods': ['method', 'methodology', 'experimental', 'materials'],
            'results': ['result', 'finding', 'observation', 'outcome'],
            'discussion': ['discussion', 'interpretation', 'analysis'],
            'conclusion': ['conclusion', 'summary', 'final']
        }

        for section_type, keywords in classification_keywords.items():
            if any(keyword in title_lower for keyword in keywords):
                return section_type

        return 'unknown'

    def _enhance_with_nlp(self, structure: DocumentStructure) -> DocumentStructure:
        """Enhance structure with NLP-based analysis"""
        if not self.nlp:
            return structure

        try:
            # Enhanced section classification
            enhanced_sections = {}

            for section_type, section_text in structure.sections.items():
                if len(section_text) > 50:
                    # NLP-based refinement
                    doc = self.nlp(section_text[:1000])  # Process first 1000 chars

                    # Extract entities and key terms
                    entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PERSON', 'GPE']]

                    # Reclassify section if confidence is low
                    refined_type = self._nlp_section_classification(section_text, section_type)

                    enhanced_sections[refined_type] = section_text

                    # Add metadata
                    if 'nlp_analysis' not in structure.metadata:
                        structure.metadata['nlp_analysis'] = {}

                    structure.metadata['nlp_analysis'][refined_type] = {
                        'entities': entities[:10],  # Top 10 entities
                        'word_count': len(doc),
                        'sentence_count': len(list(doc.sents))
                    }

            structure.sections = enhanced_sections
            return structure

        except Exception as e:
            logger.error(f"NLP enhancement failed: {e}")
            return structure

    def _nlp_section_classification(self, text: str, current_type: str) -> str:
        """Use NLP to refine section classification"""

        # Simple feature-based classification
        text_lower = text.lower()

        # Method indicators
        method_indicators = ['method', 'procedure', 'protocol', 'experiment', 'measured', 'analyzed', 'collected']
        method_score = sum(1 for indicator in method_indicators if indicator in text_lower)

        # Results indicators
        result_indicators = ['result', 'finding', 'observed', 'showed', 'demonstrated', 'figure', 'table']
        result_score = sum(1 for indicator in result_indicators if indicator in text_lower)

        # Discussion indicators
        discussion_indicators = ['discuss', 'interpret', 'suggest', 'implication', 'conclude', 'therefore']
        discussion_score = sum(1 for indicator in discussion_indicators if indicator in text_lower)

        scores = {
            'methods': method_score,
            'results': result_score,
            'discussion': discussion_score
        }

        # Return highest scoring type if significantly higher than current
        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]

        if max_score > 2 and max_type != current_type:
            return max_type

        return current_type

    def _extract_title(self, first_page_text: str) -> Optional[str]:
        """Extract paper title from first page"""
        lines = first_page_text.split('\n')

        # Look for title in first few lines
        for line in lines[:10]:
            line = line.strip()
            # Title is usually longer and contains meaningful words
            if len(line) > 20 and len(line) < 200:
                # Check if it looks like a title (not too many special chars)
                if re.match(r'^[A-Za-z0-9\s\-:,().]+$', line):
                    return line

        return None

    def _extract_abstract(self, first_page: str, second_page: str = "") -> Optional[str]:
        """Extract abstract from document text"""
        combined_text = first_page + "\n" + second_page

        # Look for abstract section
        abstract_pattern = r'(?i)abstract\s*[:\-]?\s*(.*?)(?=\n\s*(?:keywords?|introduction|1\.?\s*introduction))'
        match = re.search(abstract_pattern, combined_text, re.DOTALL)

        if match:
            abstract_text = match.group(1).strip()
            # Clean up the abstract
            abstract_text = re.sub(r'\s+', ' ', abstract_text)
            if 50 < len(abstract_text) < 2000:  # Reasonable abstract length
                return abstract_text

        return None

    def _summarize_table(self, table_data: List[List[str]]) -> str:
        """Create text summary of table data"""
        if not table_data:
            return ""

        summary_parts = []

        # Add header if available
        if table_data and table_data[0]:
            headers = [cell for cell in table_data[0] if cell]
            if headers:
                summary_parts.append(f"Table with columns: {', '.join(headers)}")

        # Add row count
        summary_parts.append(f"Contains {len(table_data)} rows")

        return ". ".join(summary_parts)

    def _merge_structures(self, base: DocumentStructure, enhanced: DocumentStructure) -> DocumentStructure:
        """Merge two document structures, preferring enhanced data"""
        merged = DocumentStructure()

        # Merge basic fields
        merged.title = enhanced.title or base.title
        merged.authors = enhanced.authors or base.authors
        merged.abstract = enhanced.abstract or base.abstract

        # Merge sections (prefer enhanced if more detailed)
        if len(enhanced.sections) > len(base.sections):
            merged.sections = enhanced.sections
        else:
            merged.sections = base.sections

        # Merge lists
        merged.figures = base.figures + enhanced.figures
        merged.tables = base.tables + enhanced.tables
        merged.references = enhanced.references or base.references

        # Merge metadata
        merged.metadata = {**base.metadata, **enhanced.metadata}

        return merged

    def _update_paper_with_structure(self, paper: Paper, structure: DocumentStructure):
        """Update Paper object with parsed structure"""
        if structure.title and not paper.title:
            paper.title = structure.title

        if structure.abstract and not paper.abstract:
            paper.abstract = structure.abstract

        # Update sections
        if structure.sections:
            paper.sections = structure.sections

        # Add parsing metadata
        if not hasattr(paper, 'parsing_metadata'):
            paper.parsing_metadata = {}

        paper.parsing_metadata.update({
            'enhanced_parsing': True,
            'sections_extracted': len(structure.sections),
            'tables_found': len(structure.tables),
            'figures_found': len(structure.figures),
            'parsing_timestamp': str(datetime.now())
        })

    def batch_parse_pdfs(self, pdf_directory: str, output_directory: str = None) -> Dict[str, Any]:
        """
        Parse multiple PDFs in a directory with section extraction

        Args:
            pdf_directory: Directory containing PDF files
            output_directory: Optional directory to save parsed results

        Returns:
            Dictionary with parsing results and statistics
        """
        if not os.path.exists(pdf_directory):
            logger.error(f"PDF directory not found: {pdf_directory}")
            return {"error": "Directory not found"}

        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]

        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_directory}")
            return {"error": "No PDF files found"}

        logger.info(f"🔄 Batch parsing {len(pdf_files)} PDFs with section extraction")

        results = {
            "total_files": len(pdf_files),
            "successful_parses": 0,
            "failed_parses": 0,
            "total_sections": 0,
            "section_distribution": {},
            "parsing_results": []
        }

        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)

            try:
                structure = self.parse_pdf_with_sections(pdf_path)

                if structure.sections:
                    results["successful_parses"] += 1
                    results["total_sections"] += len(structure.sections)

                    # Update section distribution
                    for section_type in structure.sections.keys():
                        results["section_distribution"][section_type] = results["section_distribution"].get(section_type, 0) + 1

                    # Save parsed structure if output directory specified
                    if output_directory:
                        output_file = os.path.join(output_directory, f"{Path(pdf_file).stem}_parsed.json")
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(asdict(structure), f, indent=2, default=str)

                    results["parsing_results"].append({
                        "file": pdf_file,
                        "status": "success",
                        "sections_found": len(structure.sections),
                        "has_abstract": bool(structure.abstract)
                    })

                else:
                    results["failed_parses"] += 1
                    results["parsing_results"].append({
                        "file": pdf_file,
                        "status": "failed",
                        "error": "No sections extracted"
                    })

            except Exception as e:
                results["failed_parses"] += 1
                results["parsing_results"].append({
                    "file": pdf_file,
                    "status": "error",
                    "error": str(e)
                })
                logger.error(f"Failed to parse {pdf_file}: {e}")

        success_rate = (results["successful_parses"] / results["total_files"]) * 100
        avg_sections = results["total_sections"] / max(results["successful_parses"], 1)

        logger.info(f"✅ Batch parsing complete:")
        logger.info(f"   📊 Success rate: {success_rate:.1f}% ({results['successful_parses']}/{results['total_files']})")
        logger.info(f"   📑 Total sections extracted: {results['total_sections']}")
        logger.info(f"   📈 Average sections per paper: {avg_sections:.1f}")

        return results


# Convenience functions for integration
def create_enhanced_parser() -> SciPDFParser:
    """Create enhanced PDF parser with all available components"""
    return SciPDFParser()


def parse_paper_pdf(pdf_path: str, paper: Paper = None) -> DocumentStructure:
    """Parse a single PDF with section extraction"""
    parser = SciPDFParser()
    return parser.parse_pdf_with_sections(pdf_path, paper)


if __name__ == "__main__":
    # Example usage
    parser = SciPDFParser()

    # Parse a single PDF
    structure = parser.parse_pdf_with_sections("example.pdf")
    print(f"Extracted {len(structure.sections)} sections")

    # Batch parse PDFs
    results = parser.batch_parse_pdfs("pdfs/", "parsed_results/")
    print(f"Parsed {results['successful_parses']} PDFs successfully")