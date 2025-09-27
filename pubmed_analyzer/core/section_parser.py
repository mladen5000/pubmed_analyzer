#!/usr/bin/env python3
"""
Scientific Section Parser for Section-Aware RAG
Hybrid approach combining PDFPlumber with pattern matching and NLP-based classification
"""

import logging
import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
import PyMuPDF as fitz
import regex
import spacy
from spacy.matcher import Matcher

logger = logging.getLogger(__name__)


@dataclass
class SectionContent:
    """Structured representation of a paper section"""
    section_type: str
    title: str
    content: str
    subsections: List[Dict[str, str]]
    page_numbers: List[int]
    confidence_score: float
    citations: List[str]
    figures_tables: List[str]


@dataclass
class ParsedPaper:
    """Complete parsed paper with all sections"""
    title: str
    authors: List[str]
    abstract: Optional[SectionContent]
    sections: Dict[str, SectionContent]
    references: List[str]
    parsing_quality: float
    structure_confidence: float


class ScientificSectionParser:
    """
    Advanced scientific paper section parser using hybrid approach:
    1. Pattern-based section detection
    2. NLP-based content classification
    3. Structure validation and confidence scoring
    """

    def __init__(self):
        self.section_patterns = self._initialize_section_patterns()
        self.citation_patterns = self._initialize_citation_patterns()
        self.figure_table_patterns = self._initialize_figure_table_patterns()

        # Try to load spaCy model, fallback gracefully
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Section classification will be pattern-based only.")
            self.nlp = None

        self.matcher = None
        if self.nlp:
            self.matcher = Matcher(self.nlp.vocab)
            self._setup_section_matchers()

    def _initialize_section_patterns(self) -> Dict[str, List[str]]:
        """Initialize comprehensive section detection patterns"""
        return {
            'abstract': [
                r'(?i)^(?:\d+\.?\s*)?abstract\s*$',
                r'(?i)^(?:\d+\.?\s*)?summary\s*$',
                r'(?i)^(?:\d+\.?\s*)?overview\s*$'
            ],
            'introduction': [
                r'(?i)^(?:\d+\.?\s*)?introduction\s*$',
                r'(?i)^(?:\d+\.?\s*)?background\s*$',
                r'(?i)^(?:\d+\.?\s*)?motivation\s*$',
                r'(?i)^(?:\d+\.?\s*)?rationale\s*$'
            ],
            'methods': [
                r'(?i)^(?:\d+\.?\s*)?methods?\s*$',
                r'(?i)^(?:\d+\.?\s*)?methodology\s*$',
                r'(?i)^(?:\d+\.?\s*)?materials?\s+and\s+methods?\s*$',
                r'(?i)^(?:\d+\.?\s*)?experimental\s+(?:design|procedures?|setup)\s*$',
                r'(?i)^(?:\d+\.?\s*)?approach\s*$'
            ],
            'results': [
                r'(?i)^(?:\d+\.?\s*)?results?\s*$',
                r'(?i)^(?:\d+\.?\s*)?findings?\s*$',
                r'(?i)^(?:\d+\.?\s*)?observations?\s*$',
                r'(?i)^(?:\d+\.?\s*)?outcomes?\s*$'
            ],
            'discussion': [
                r'(?i)^(?:\d+\.?\s*)?discussion\s*$',
                r'(?i)^(?:\d+\.?\s*)?analysis\s*$',
                r'(?i)^(?:\d+\.?\s*)?interpretation\s*$',
                r'(?i)^(?:\d+\.?\s*)?implications?\s*$'
            ],
            'conclusion': [
                r'(?i)^(?:\d+\.?\s*)?conclusions?\s*$',
                r'(?i)^(?:\d+\.?\s*)?concluding\s+remarks?\s*$',
                r'(?i)^(?:\d+\.?\s*)?final\s+thoughts?\s*$',
                r'(?i)^(?:\d+\.?\s*)?summary\s+and\s+conclusions?\s*$'
            ],
            'references': [
                r'(?i)^(?:\d+\.?\s*)?references?\s*$',
                r'(?i)^(?:\d+\.?\s*)?bibliography\s*$',
                r'(?i)^(?:\d+\.?\s*)?works?\s+cited\s*$',
                r'(?i)^(?:\d+\.?\s*)?literature\s+cited\s*$'
            ],
            'acknowledgments': [
                r'(?i)^(?:\d+\.?\s*)?acknowledgments?\s*$',
                r'(?i)^(?:\d+\.?\s*)?acknowledgements?\s*$',
                r'(?i)^(?:\d+\.?\s*)?funding\s*$',
                r'(?i)^(?:\d+\.?\s*)?grants?\s*$'
            ]
        }

    def _initialize_citation_patterns(self) -> List[str]:
        """Initialize citation detection patterns"""
        return [
            r'\[(\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*)\]',  # [1], [1-3], [1,2,3]
            r'\(([A-Za-z]+(?:\s+et\s+al\.?)?,?\s+\d{4}[a-z]?(?:;\s*[A-Za-z]+(?:\s+et\s+al\.?)?,?\s+\d{4}[a-z]?)*)\)',  # (Author, 2020)
            r'(?:^|\s)([A-Za-z]+(?:\s+et\s+al\.?)?\s+\(\d{4}[a-z]?\))',  # Author (2020)
            r'\b(doi:\s*10\.\d+/[^\s]+)',  # DOI
            r'\b(PMID:\s*\d+)',  # PMID
        ]

    def _initialize_figure_table_patterns(self) -> List[str]:
        """Initialize figure and table detection patterns"""
        return [
            r'(?i)figure\s+(\d+[a-z]?)',
            r'(?i)fig\.?\s+(\d+[a-z]?)',
            r'(?i)table\s+(\d+[a-z]?)',
            r'(?i)tab\.?\s+(\d+[a-z]?)',
            r'(?i)supplementary\s+(?:figure|table)\s+(\d+[a-z]?)',
            r'(?i)(?:figure|table)\s+s(\d+[a-z]?)'
        ]

    def _setup_section_matchers(self):
        """Setup spaCy matchers for section detection"""
        if not self.matcher:
            return

        # Add patterns for section headers
        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                # Convert regex to spaCy pattern (simplified)
                pattern_tokens = [{"TEXT": {"REGEX": pattern}}]
                self.matcher.add(f"SECTION_{section_type.upper()}", [pattern_tokens])

    def parse_pdf_sections(self, pdf_path: str) -> ParsedPaper:
        """
        Main entry point for parsing PDF into structured sections

        Args:
            pdf_path: Path to the PDF file

        Returns:
            ParsedPaper object with structured content
        """
        logger.info(f"🔍 Parsing PDF sections: {Path(pdf_path).name}")

        try:
            # Extract text and structure information
            pdf_text, page_info = self._extract_pdf_text(pdf_path)

            if not pdf_text.strip():
                logger.error(f"❌ No text extracted from PDF: {pdf_path}")
                return self._create_empty_parsed_paper("No text extracted")

            # Detect sections using hybrid approach
            sections = self._detect_sections(pdf_text, page_info)

            # Extract metadata
            title, authors = self._extract_metadata(pdf_text)

            # Parse references
            references = self._extract_references(pdf_text)

            # Calculate parsing quality
            parsing_quality = self._calculate_parsing_quality(sections, pdf_text)
            structure_confidence = self._calculate_structure_confidence(sections)

            parsed_paper = ParsedPaper(
                title=title,
                authors=authors,
                abstract=sections.get('abstract'),
                sections={k: v for k, v in sections.items() if k != 'abstract'},
                references=references,
                parsing_quality=parsing_quality,
                structure_confidence=structure_confidence
            )

            logger.info(f"✅ Parsed {len(sections)} sections with {parsing_quality:.2f} quality score")
            return parsed_paper

        except Exception as e:
            logger.error(f"❌ Error parsing PDF {pdf_path}: {e}")
            return self._create_empty_parsed_paper(str(e))

    def _extract_pdf_text(self, pdf_path: str) -> Tuple[str, List[Dict]]:
        """Extract text and page information from PDF"""
        try:
            # Try PDFPlumber first for better structure preservation
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                page_info = []

                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        full_text += f"\n--- PAGE {page_num} ---\n{page_text}\n"
                        page_info.append({
                            'page_number': page_num,
                            'text': page_text,
                            'text_length': len(page_text)
                        })

                if full_text.strip():
                    return full_text, page_info

        except Exception as e:
            logger.warning(f"PDFPlumber failed for {pdf_path}: {e}, trying PyMuPDF")

        # Fallback to PyMuPDF
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            page_info = []

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                full_text += f"\n--- PAGE {page_num + 1} ---\n{page_text}\n"
                page_info.append({
                    'page_number': page_num + 1,
                    'text': page_text,
                    'text_length': len(page_text)
                })

            doc.close()
            return full_text, page_info

        except Exception as e:
            logger.error(f"Both PDF extraction methods failed for {pdf_path}: {e}")
            return "", []

    def _detect_sections(self, text: str, page_info: List[Dict]) -> Dict[str, SectionContent]:
        """Detect and extract sections using hybrid approach"""
        sections = {}
        lines = text.split('\n')
        current_section = None
        current_content = []
        current_page = 1

        # Track page boundaries
        page_boundaries = {}
        for line_num, line in enumerate(lines):
            if '--- PAGE' in line:
                page_match = re.search(r'--- PAGE (\d+) ---', line)
                if page_match:
                    page_boundaries[line_num] = int(page_match.group(1))

        for line_num, line in enumerate(lines):
            line = line.strip()

            # Update current page
            if line_num in page_boundaries:
                current_page = page_boundaries[line_num]
                continue

            if not line or '--- PAGE' in line:
                continue

            # Check if line is a section header
            detected_section = self._classify_section_header(line)

            if detected_section:
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = self._create_section_content(
                        current_section,
                        '\n'.join(current_content),
                        [current_page]
                    )

                # Start new section
                current_section = detected_section
                current_content = []
            elif current_section:
                current_content.append(line)

        # Save final section
        if current_section and current_content:
            sections[current_section] = self._create_section_content(
                current_section,
                '\n'.join(current_content),
                [current_page]
            )

        return sections

    def _classify_section_header(self, line: str) -> Optional[str]:
        """Classify a line as a section header"""
        line_clean = line.strip()

        # Try pattern matching first
        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                if regex.match(pattern, line_clean):
                    return section_type

        # Additional heuristics for common variations
        line_lower = line_clean.lower()

        # Check for numbered sections
        if re.match(r'^\d+\.?\s*', line_clean):
            line_content = re.sub(r'^\d+\.?\s*', '', line_lower)
            for section_type, patterns in self.section_patterns.items():
                if any(section_type in line_content for section_type in [section_type]):
                    return section_type

        return None

    def _create_section_content(self, section_type: str, content: str, pages: List[int]) -> SectionContent:
        """Create structured section content with metadata"""

        # Extract citations
        citations = self._extract_citations_from_text(content)

        # Extract figure/table references
        figures_tables = self._extract_figure_table_refs(content)

        # Simple subsection detection (lines that start with letters or numbers)
        subsections = []
        subsection_pattern = r'^([A-Za-z0-9]+\.?\s+[A-Z][^.]+\.?)$'
        for line in content.split('\n'):
            line = line.strip()
            if re.match(subsection_pattern, line) and len(line) < 100:
                subsections.append({
                    'title': line,
                    'content': ''  # Would need more sophisticated extraction
                })

        # Calculate confidence based on section characteristics
        confidence = self._calculate_section_confidence(section_type, content)

        return SectionContent(
            section_type=section_type,
            title=section_type.title(),
            content=content,
            subsections=subsections,
            page_numbers=pages,
            confidence_score=confidence,
            citations=citations,
            figures_tables=figures_tables
        )

    def _extract_citations_from_text(self, text: str) -> List[str]:
        """Extract citations from text using multiple patterns"""
        citations = []

        for pattern in self.citation_patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)

        # Deduplicate and clean
        return list(set([cite for cite in citations if cite.strip()]))

    def _extract_figure_table_refs(self, text: str) -> List[str]:
        """Extract figure and table references"""
        refs = []

        for pattern in self.figure_table_patterns:
            matches = re.findall(pattern, text)
            refs.extend(matches)

        return list(set(refs))

    def _extract_metadata(self, text: str) -> Tuple[str, List[str]]:
        """Extract title and authors from paper text"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # Simple heuristic: title is usually one of the first few lines
        title = "Unknown Title"
        authors = []

        # Look for title in first 10 lines
        for line in lines[:10]:
            if len(line) > 20 and not line.lower().startswith(('abstract', 'introduction', 'page')):
                if not re.match(r'^\d+\.', line):  # Skip numbered sections
                    title = line
                    break

        # Look for author patterns (simplified)
        author_patterns = [
            r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # FirstName LastName
            r'([A-Z]\.\s*[A-Z][a-z]+)',  # F. LastName
        ]

        for line in lines[:20]:
            for pattern in author_patterns:
                matches = re.findall(pattern, line)
                authors.extend(matches)

        return title, list(set(authors))

    def _extract_references(self, text: str) -> List[str]:
        """Extract reference list from paper"""
        lines = text.split('\n')
        references = []
        in_references = False

        for line in lines:
            line = line.strip()

            # Check if we've entered references section
            if any(re.match(pattern, line) for pattern in self.section_patterns['references']):
                in_references = True
                continue

            if in_references and line:
                # Simple reference detection (lines starting with numbers or authors)
                if re.match(r'^\d+\.?\s+', line) or re.match(r'^[A-Z][a-z]+', line):
                    references.append(line)

        return references

    def _calculate_section_confidence(self, section_type: str, content: str) -> float:
        """Calculate confidence score for section classification"""
        base_score = 0.7

        # Boost confidence based on content characteristics
        content_lower = content.lower()

        confidence_boosters = {
            'abstract': ['objective', 'background', 'methods', 'results', 'conclusions'],
            'introduction': ['background', 'previous', 'prior', 'literature', 'motivation'],
            'methods': ['procedure', 'protocol', 'experimental', 'measurement', 'analysis'],
            'results': ['showed', 'demonstrated', 'observed', 'found', 'significant'],
            'discussion': ['interpret', 'implication', 'limitation', 'future', 'conclusion'],
        }

        if section_type in confidence_boosters:
            boost_words = confidence_boosters[section_type]
            boost_count = sum(1 for word in boost_words if word in content_lower)
            boost_score = min(0.25, boost_count * 0.05)
            base_score += boost_score

        # Penalize very short sections
        if len(content) < 100:
            base_score -= 0.2

        return min(1.0, max(0.0, base_score))

    def _calculate_parsing_quality(self, sections: Dict, text: str) -> float:
        """Calculate overall parsing quality score"""
        if not sections:
            return 0.0

        # Base score for having sections
        quality = 0.3

        # Boost for having key sections
        important_sections = ['abstract', 'introduction', 'methods', 'results']
        found_important = sum(1 for section in important_sections if section in sections)
        quality += (found_important / len(important_sections)) * 0.4

        # Boost for section content quality
        avg_confidence = sum(section.confidence_score for section in sections.values()) / len(sections)
        quality += avg_confidence * 0.3

        return min(1.0, quality)

    def _calculate_structure_confidence(self, sections: Dict) -> float:
        """Calculate confidence in document structure detection"""
        if not sections:
            return 0.0

        # Check for standard academic paper structure
        expected_order = ['abstract', 'introduction', 'methods', 'results', 'discussion']
        found_sections = list(sections.keys())

        # Simple order checking
        order_score = 0.0
        for i, expected in enumerate(expected_order):
            if expected in found_sections:
                actual_position = found_sections.index(expected)
                # Closer to expected position = higher score
                position_penalty = abs(actual_position - i) / len(found_sections)
                order_score += (1.0 - position_penalty) / len(expected_order)

        return min(1.0, order_score)

    def _create_empty_parsed_paper(self, error_msg: str) -> ParsedPaper:
        """Create empty parsed paper for error cases"""
        return ParsedPaper(
            title="Parse Error",
            authors=[],
            abstract=None,
            sections={},
            references=[],
            parsing_quality=0.0,
            structure_confidence=0.0
        )

    def extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """Extract detailed citation information"""
        citations = []

        for pattern in self.citation_patterns:
            for match in re.finditer(pattern, text):
                citation_text = match.group(0)
                start_pos = match.start()
                end_pos = match.end()

                # Extract context (surrounding text)
                context_start = max(0, start_pos - 100)
                context_end = min(len(text), end_pos + 100)
                context = text[context_start:context_end]

                citations.append({
                    'text': citation_text,
                    'position': (start_pos, end_pos),
                    'context': context,
                    'type': self._classify_citation_type(citation_text)
                })

        return citations

    def _classify_citation_type(self, citation: str) -> str:
        """Classify citation format type"""
        if citation.startswith('[') and citation.endswith(']'):
            return 'numbered'
        elif '(' in citation and ')' in citation:
            return 'author_year'
        elif 'doi:' in citation.lower():
            return 'doi'
        elif 'pmid:' in citation.lower():
            return 'pmid'
        else:
            return 'unknown'

    def identify_figures_tables(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract figure and table metadata from PDF"""
        figures_tables = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract tables
                    tables = page.extract_tables()
                    for i, table in enumerate(tables):
                        figures_tables.append({
                            'type': 'table',
                            'page': page_num,
                            'index': i,
                            'data': table,
                            'caption': self._extract_table_caption(page, table)
                        })

                    # Look for figure references in text
                    page_text = page.extract_text()
                    if page_text:
                        for pattern in self.figure_table_patterns:
                            matches = re.finditer(pattern, page_text, re.IGNORECASE)
                            for match in matches:
                                figures_tables.append({
                                    'type': 'figure_reference',
                                    'page': page_num,
                                    'text': match.group(0),
                                    'position': match.span()
                                })

        except Exception as e:
            logger.warning(f"Error extracting figures/tables from {pdf_path}: {e}")

        return figures_tables

    def _extract_table_caption(self, page, table) -> str:
        """Extract caption for a table (simplified implementation)"""
        # This is a simplified implementation
        # In practice, you'd want more sophisticated caption detection
        page_text = page.extract_text()
        if page_text:
            # Look for "Table X" patterns near the table
            table_patterns = [r'Table\s+\d+[:.]\s*([^\n]+)', r'Tab\.\s+\d+[:.]\s*([^\n]+)']
            for pattern in table_patterns:
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    return match.group(1)
        return ""