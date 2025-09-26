from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime


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
    sections: Dict[str, str] = field(default_factory=dict)

    # File paths
    pdf_path: Optional[str] = None
    txt_path: Optional[str] = None
    markdown_path: Optional[str] = None

    # Processing status
    download_success: bool = False
    processing_success: bool = False
    error_message: Optional[str] = None

    # Analysis results
    embeddings: Optional[Any] = None
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