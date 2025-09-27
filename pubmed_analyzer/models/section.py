#!/usr/bin/env python3
"""
Enhanced Section Data Models for RAG System
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Union
from datetime import datetime
from enum import Enum
import uuid


class SectionType(Enum):
    """Standardized section types for biomedical papers"""
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    BACKGROUND = "background"
    METHODS = "methods"
    METHODOLOGY = "methodology"
    MATERIALS_METHODS = "materials_methods"
    RESULTS = "results"
    FINDINGS = "findings"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    CONCLUSIONS = "conclusions"
    LIMITATIONS = "limitations"
    FUTURE_WORK = "future_work"
    ACKNOWLEDGMENTS = "acknowledgments"
    REFERENCES = "references"
    SUPPLEMENTARY = "supplementary"
    APPENDIX = "appendix"
    CASE_STUDY = "case_study"
    RELATED_WORK = "related_work"
    LITERATURE_REVIEW = "literature_review"
    OTHER = "other"

    @classmethod
    def from_text(cls, text: str) -> 'SectionType':
        """Map section text to standardized type"""
        text_lower = text.lower().strip()

        # Exact matches
        for section_type in cls:
            if text_lower == section_type.value:
                return section_type

        # Fuzzy matches
        if any(keyword in text_lower for keyword in ["intro", "introduction"]):
            return cls.INTRODUCTION
        elif any(keyword in text_lower for keyword in ["method", "methodology", "materials"]):
            return cls.METHODS
        elif any(keyword in text_lower for keyword in ["result", "findings"]):
            return cls.RESULTS
        elif any(keyword in text_lower for keyword in ["discuss", "discussion"]):
            return cls.DISCUSSION
        elif any(keyword in text_lower for keyword in ["conclusion", "concluding"]):
            return cls.CONCLUSION
        elif any(keyword in text_lower for keyword in ["abstract", "summary"]):
            return cls.ABSTRACT
        elif any(keyword in text_lower for keyword in ["background", "related work", "literature"]):
            return cls.BACKGROUND
        elif any(keyword in text_lower for keyword in ["limitation", "limitations"]):
            return cls.LIMITATIONS
        elif any(keyword in text_lower for keyword in ["future", "next steps"]):
            return cls.FUTURE_WORK
        elif any(keyword in text_lower for keyword in ["case study", "case report"]):
            return cls.CASE_STUDY
        else:
            return cls.OTHER


@dataclass
class SectionMetadata:
    """Enhanced metadata for paper sections"""
    section_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    paper_pmid: str = ""
    section_type: SectionType = SectionType.OTHER
    section_title: str = ""
    section_order: int = 0

    # Content characteristics
    word_count: int = 0
    char_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0

    # Semantic metadata
    key_topics: List[str] = field(default_factory=list)
    named_entities: Dict[str, List[str]] = field(default_factory=dict)
    complexity_score: float = 0.0
    readability_score: float = 0.0

    # Technical metadata
    has_citations: bool = False
    citation_count: int = 0
    has_figures: bool = False
    figure_count: int = 0
    has_tables: bool = False
    table_count: int = 0

    # Processing metadata
    extraction_method: str = ""
    confidence_score: float = 1.0
    processing_timestamp: datetime = field(default_factory=datetime.now)
    embedding_model: Optional[str] = None

    def __post_init__(self):
        """Validate and normalize metadata"""
        if isinstance(self.section_type, str):
            self.section_type = SectionType.from_text(self.section_type)


@dataclass
class ProcessedSection:
    """A fully processed paper section with content and metadata"""
    content: str
    metadata: SectionMetadata

    # Text processing results
    cleaned_content: str = ""
    tokenized_content: List[str] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)
    paragraphs: List[str] = field(default_factory=list)

    # Embeddings (multiple strategies)
    dense_embedding: Optional[List[float]] = None
    sparse_embedding: Optional[Dict[str, float]] = None
    section_specific_embedding: Optional[List[float]] = None

    # Relationships
    related_sections: List[str] = field(default_factory=list)  # Section IDs
    cross_references: List[Dict[str, Any]] = field(default_factory=list)

    # Quality metrics
    information_density: float = 0.0
    novelty_score: float = 0.0
    relevance_indicators: Dict[str, float] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if section has valid content"""
        return (
            len(self.content.strip()) > 20 and
            self.metadata.word_count > 5 and
            self.metadata.confidence_score > 0.3
        )

    @property
    def section_id(self) -> str:
        """Get section ID from metadata"""
        return self.metadata.section_id

    @property
    def section_type(self) -> SectionType:
        """Get section type from metadata"""
        return self.metadata.section_type


@dataclass
class SectionCollection:
    """Collection of sections for a single paper"""
    paper_pmid: str
    sections: List[ProcessedSection] = field(default_factory=list)

    # Paper-level section analysis
    section_distribution: Dict[SectionType, int] = field(default_factory=dict)
    total_sections: int = 0
    total_word_count: int = 0

    # Inter-section relationships
    section_graph: Dict[str, List[str]] = field(default_factory=dict)
    narrative_flow_score: float = 0.0
    structural_completeness: float = 0.0

    def __post_init__(self):
        """Calculate derived metrics"""
        self.total_sections = len(self.sections)
        self.total_word_count = sum(s.metadata.word_count for s in self.sections)

        # Calculate section distribution
        for section in self.sections:
            section_type = section.section_type
            self.section_distribution[section_type] = self.section_distribution.get(section_type, 0) + 1

    def get_sections_by_type(self, section_type: SectionType) -> List[ProcessedSection]:
        """Get all sections of a specific type"""
        return [s for s in self.sections if s.section_type == section_type]

    def get_section_by_id(self, section_id: str) -> Optional[ProcessedSection]:
        """Get section by ID"""
        for section in self.sections:
            if section.section_id == section_id:
                return section
        return None

    def has_section_type(self, section_type: SectionType) -> bool:
        """Check if collection has sections of given type"""
        return section_type in self.section_distribution

    @property
    def available_section_types(self) -> List[SectionType]:
        """Get list of available section types"""
        return list(self.section_distribution.keys())


@dataclass
class QueryContext:
    """Context for section-aware queries"""
    query: str
    target_section_types: List[SectionType] = field(default_factory=list)
    paper_filters: Dict[str, Any] = field(default_factory=dict)

    # Query preferences
    prefer_recent: bool = False
    prefer_high_impact: bool = False
    require_full_text: bool = False

    # Retrieval parameters
    max_sections: int = 10
    similarity_threshold: float = 0.5
    cross_section_context: bool = True

    # Advanced options
    semantic_expansion: bool = True
    query_decomposition: bool = False
    multi_hop_retrieval: bool = False