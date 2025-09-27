from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Union
from datetime import datetime
import json


@dataclass
class Paper:
    """Data model for a scientific paper with metadata from multiple NCBI services"""

    # Primary identifiers
    pmid: str
    pmcid: Optional[str] = None
    doi: Optional[str] = None

    # Basic metadata (from E-utilities)
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    journal: Optional[str] = None
    pub_date: Optional[datetime] = None
    abstract: Optional[str] = None

    # PMC-specific metadata
    pmc_metadata: Optional[Dict[str, Any]] = None
    license: Optional[str] = None
    is_retracted: bool = False

    # Full-text content
    has_fulltext: bool = False
    full_text: Optional[str] = None
    sections: Dict[str, str] = field(default_factory=dict)  # Legacy simple sections

    # Section-aware enhancements
    structured_sections: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    section_hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    figures_tables: List[Dict[str, Any]] = field(default_factory=list)
    author_affiliations: List[Dict[str, Any]] = field(default_factory=list)

    # Quality metrics for section parsing
    section_parsing_quality: float = 0.0
    structure_confidence: float = 0.0

    # Analysis metadata
    embedding_models_used: List[str] = field(default_factory=list)
    chromadb_ids: Dict[str, str] = field(default_factory=dict)  # section_type -> document_id
    section_embeddings: Dict[str, Any] = field(default_factory=dict)

    # File paths
    pdf_path: Optional[str] = None
    txt_path: Optional[str] = None
    markdown_path: Optional[str] = None

    # Processing status
    download_success: bool = False
    processing_success: bool = False
    section_parsing_success: bool = False
    error_message: Optional[str] = None

    # Analysis results
    embeddings: Optional[Any] = None  # Legacy embeddings
    keywords: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate and normalize data after initialization"""
        if self.pmid and not self.pmid.startswith(('PMID:', 'pmid:')):
            self.pmid = f"PMID:{self.pmid}"

        if self.pmcid and not self.pmcid.startswith(('PMC', 'pmc')):
            self.pmcid = f"PMC{self.pmcid}"

    def __repr__(self):
        """Clean representation without verbose fields"""
        return f"Paper(pmid='{self.pmid}', pmcid={self.pmcid})"

    @property
    def clean_pmid(self) -> str:
        """Return PMID without prefix"""
        return self.pmid.replace('PMID:', '').replace('pmid:', '')

    @property
    def clean_pmcid(self) -> Optional[str]:
        """Return PMC ID without prefix"""
        if self.pmcid:
            return self.pmcid.replace('PMC', '').replace('pmc', '')
        return None

    @property
    def has_pdf(self) -> bool:
        """Check if PDF file exists and is accessible"""
        import os
        return self.pdf_path and os.path.exists(self.pdf_path)

    @property
    def has_text(self) -> bool:
        """Check if text file exists and is accessible"""
        import os
        return self.txt_path and os.path.exists(self.txt_path)

    @property
    def year(self) -> Optional[int]:
        """Extract year from publication date"""
        if self.pub_date:
            return self.pub_date.year
        return None

    # Section-aware properties and methods

    @property
    def has_structured_sections(self) -> bool:
        """Check if paper has been parsed into structured sections"""
        return bool(self.structured_sections) and self.section_parsing_success

    @property
    def available_sections(self) -> List[str]:
        """Get list of available section types"""
        return list(self.structured_sections.keys())

    @property
    def section_count(self) -> int:
        """Get total number of parsed sections"""
        return len(self.structured_sections)

    def get_section_content(self, section_type: str) -> Optional[str]:
        """Get content for a specific section type"""
        if section_type in self.structured_sections:
            return self.structured_sections[section_type].get('content')
        # Fallback to legacy sections
        return self.sections.get(section_type)

    def get_section_metadata(self, section_type: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific section"""
        return self.structured_sections.get(section_type)

    def add_structured_section(self, section_type: str, content: str,
                             metadata: Optional[Dict[str, Any]] = None):
        """Add a structured section with metadata"""
        section_data = {
            'content': content,
            'content_length': len(content),
            'citations': [],
            'figures_tables': [],
            'subsections': [],
            'confidence_score': 0.0,
            'page_numbers': [],
        }

        if metadata:
            section_data.update(metadata)

        self.structured_sections[section_type] = section_data

    def get_citations_in_section(self, section_type: str) -> List[Dict[str, Any]]:
        """Get citations found in a specific section"""
        section = self.structured_sections.get(section_type, {})
        return section.get('citations', [])

    def get_figures_tables_in_section(self, section_type: str) -> List[str]:
        """Get figure/table references in a specific section"""
        section = self.structured_sections.get(section_type, {})
        return section.get('figures_tables', [])

    def get_section_quality_score(self, section_type: str) -> float:
        """Get quality/confidence score for a specific section"""
        section = self.structured_sections.get(section_type, {})
        return section.get('confidence_score', 0.0)

    def has_section_type(self, section_type: str) -> bool:
        """Check if paper has a specific section type"""
        return section_type in self.structured_sections

    def get_sections_by_quality(self, min_quality: float = 0.7) -> List[str]:
        """Get sections with quality score above threshold"""
        return [
            section_type for section_type, data in self.structured_sections.items()
            if data.get('confidence_score', 0.0) >= min_quality
        ]

    def add_citation(self, citation_data: Dict[str, Any]):
        """Add citation with context information"""
        self.citations.append(citation_data)

    def add_figure_table(self, figure_table_data: Dict[str, Any]):
        """Add figure or table metadata"""
        self.figures_tables.append(figure_table_data)

    def get_section_embedding_id(self, section_type: str) -> Optional[str]:
        """Get ChromaDB document ID for a section"""
        return self.chromadb_ids.get(section_type)

    def set_section_embedding_id(self, section_type: str, document_id: str):
        """Set ChromaDB document ID for a section"""
        self.chromadb_ids[section_type] = document_id

    def get_research_methodology(self) -> Optional[str]:
        """Extract research methodology from methods section"""
        methods_content = self.get_section_content('methods')
        if methods_content:
            # Simple heuristic to identify methodology type
            content_lower = methods_content.lower()
            if any(word in content_lower for word in ['experiment', 'trial', 'study design']):
                return 'experimental'
            elif any(word in content_lower for word in ['survey', 'questionnaire', 'interview']):
                return 'survey'
            elif any(word in content_lower for word in ['review', 'systematic', 'meta-analysis']):
                return 'review'
            elif any(word in content_lower for word in ['simulation', 'model', 'computational']):
                return 'computational'
            else:
                return 'unknown'
        return None

    def get_key_findings(self) -> List[str]:
        """Extract key findings from results section"""
        results_content = self.get_section_content('results')
        if results_content:
            # Simple extraction of sentences with finding indicators
            sentences = results_content.split('.')
            findings = []
            finding_indicators = ['showed', 'demonstrated', 'found', 'observed', 'revealed',
                                'indicated', 'significant', 'increased', 'decreased']

            for sentence in sentences:
                if any(indicator in sentence.lower() for indicator in finding_indicators):
                    findings.append(sentence.strip())

            return findings[:5]  # Return top 5 findings
        return []

    def get_limitations(self) -> List[str]:
        """Extract limitations mentioned in discussion or conclusion"""
        limitations = []

        for section_type in ['discussion', 'conclusion']:
            content = self.get_section_content(section_type)
            if content:
                sentences = content.split('.')
                limitation_indicators = ['limitation', 'limit', 'constraint', 'drawback',
                                       'shortcoming', 'weakness', 'caveat']

                for sentence in sentences:
                    if any(indicator in sentence.lower() for indicator in limitation_indicators):
                        limitations.append(sentence.strip())

        return limitations[:3]  # Return top 3 limitations

    def to_section_summary(self) -> Dict[str, Any]:
        """Create a summary of section structure and quality"""
        return {
            'pmid': self.pmid,
            'title': self.title,
            'has_structured_sections': self.has_structured_sections,
            'section_count': self.section_count,
            'available_sections': self.available_sections,
            'parsing_quality': self.section_parsing_quality,
            'structure_confidence': self.structure_confidence,
            'citation_count': len(self.citations),
            'figures_tables_count': len(self.figures_tables),
            'section_qualities': {
                section_type: data.get('confidence_score', 0.0)
                for section_type, data in self.structured_sections.items()
            },
            'research_methodology': self.get_research_methodology(),
            'has_embeddings': bool(self.chromadb_ids)
        }

    def export_sections_json(self) -> str:
        """Export structured sections to JSON string"""
        export_data = {
            'paper_metadata': {
                'pmid': self.pmid,
                'pmcid': self.pmcid,
                'title': self.title,
                'authors': self.authors,
                'journal': self.journal,
                'year': self.year
            },
            'structured_sections': self.structured_sections,
            'citations': self.citations,
            'figures_tables': self.figures_tables,
            'quality_metrics': {
                'section_parsing_quality': self.section_parsing_quality,
                'structure_confidence': self.structure_confidence
            }
        }
        return json.dumps(export_data, indent=2, default=str)