"""
PubMed Analyzer Package
Modular scientific literature analysis pipeline with full-text PDF processing,
RAG, vector search, and advanced analytics.
"""

from .core.search import PubMedSearcher
from .core.id_converter import PMIDToPMCConverter
from .core.pdf_fetcher import UnifiedPDFFetcher
from .core.nlp_analyzer import AdvancedNLPAnalyzer
from .core.rag_analyzer import EnhancedRAGAnalyzer
from .models.paper import Paper
from .utils.ncbi_client import NCBIClient
from .utils.validators import PDFValidator

__version__ = "2.0.0"
__author__ = "Enhanced with advanced NLP/ML and RAG capabilities"

__all__ = [
    'PubMedSearcher',
    'PMIDToPMCConverter',
    'UnifiedPDFFetcher',
    'AdvancedNLPAnalyzer',
    'EnhancedRAGAnalyzer',
    'Paper',
    'NCBIClient',
    'PDFValidator'
]