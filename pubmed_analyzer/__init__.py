"""
PubMed Analyzer Package
Modular scientific literature analysis pipeline with full-text PDF processing,
RAG, vector search, and advanced analytics.
"""

from .core.search import PubMedSearcher
from .core.id_converter import PMIDToPMCConverter
from .core.pdf_fetcher import UnifiedPDFFetcher
from .models.paper import Paper
from .utils.ncbi_client import NCBIClient
from .utils.validators import PDFValidator

__version__ = "1.0.0"
__author__ = "Enhanced for modular architecture"

__all__ = [
    'PubMedSearcher',
    'PMIDToPMCConverter',
    'UnifiedPDFFetcher',
    'Paper',
    'NCBIClient',
    'PDFValidator'
]