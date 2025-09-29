#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for the section-aware RAG test suite
"""

import pytest
import tempfile
import shutil
import logging
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress noisy loggers during tests
logging.getLogger('chromadb').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data"""
    temp_dir = tempfile.mkdtemp(prefix="pubmed_analyzer_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_environment():
    """Mock environment variables for testing"""
    test_env = {
        'PUBMED_EMAIL': 'test@example.com',
        'OPENAI_API_KEY': 'test_openai_key',
        'DEEPSEEK_API_KEY': 'test_deepseek_key',
        'NCBI_API_KEY': 'test_ncbi_key'
    }

    with patch.dict(os.environ, test_env):
        yield test_env


@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB for tests that don't need real vector operations"""
    mock_client = Mock()
    mock_collection = Mock()

    # Mock collection methods
    mock_collection.add.return_value = None
    mock_collection.query.return_value = {
        'documents': [['Sample document content']],
        'metadatas': [[{'pmid': 'TEST001', 'section_type': 'abstract'}]],
        'distances': [[0.1]]
    }
    mock_collection.get.return_value = {
        'documents': ['Sample content'],
        'metadatas': [{'pmid': 'TEST001'}],
        'ids': ['test_id']
    }
    mock_collection.count.return_value = 1
    mock_collection.delete.return_value = None

    # Mock client methods
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client.get_collection.return_value = mock_collection

    with patch('chromadb.PersistentClient', return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_embeddings():
    """Generate mock embeddings for testing"""
    def generate_embedding(dim=384):
        return np.random.rand(dim)

    def generate_batch_embeddings(batch_size=5, dim=384):
        return np.random.rand(batch_size, dim)

    return {
        'single': generate_embedding,
        'batch': generate_batch_embeddings,
        'section_embeddings': {
            'abstract': generate_embedding(),
            'introduction': generate_embedding(),
            'methods': generate_embedding(),
            'results': generate_embedding(),
            'discussion': generate_embedding(),
            'conclusion': generate_embedding()
        }
    }


@pytest.fixture
def sample_papers():
    """Comprehensive sample paper data for testing"""
    return [
        {
            'pmid': 'TEST_001',
            'pmcid': 'PMC_TEST_001',
            'title': 'Machine Learning Applications in Biomedical Research: A Systematic Review',
            'authors': ['Smith, John A.', 'Johnson, Mary B.', 'Chen, Li'],
            'journal': 'Nature Biotechnology',
            'year': 2023,
            'doi': '10.1038/test.001',
            'abstract': """
            Background: Machine learning has revolutionized biomedical research by enabling automated analysis of complex biological data.
            Objective: To systematically review machine learning applications in biomedical research from 2020-2023.
            Methods: We searched PubMed, IEEE Xplore, and ACM Digital Library using terms related to machine learning and biomedicine.
            Results: We identified 1,247 relevant studies. Deep learning was the most common approach (65.3%), followed by ensemble methods (23.1%).
            Conclusions: Machine learning shows significant promise for advancing biomedical research, particularly in drug discovery and personalized medicine.
            """,
            'sections': {
                'Introduction': """
                The integration of machine learning (ML) into biomedical research has accelerated dramatically over the past decade.
                Traditional computational approaches in biology and medicine are increasingly being augmented or replaced by ML algorithms
                that can identify patterns in complex, high-dimensional datasets. This transformation is particularly evident in areas
                such as genomics, medical imaging, drug discovery, and clinical decision support systems.
                """,
                'Methods': """
                We conducted a systematic literature review following PRISMA guidelines. Our search strategy included three major databases:
                PubMed (MEDLINE), IEEE Xplore, and ACM Digital Library. Search terms combined concepts of machine learning
                ("machine learning", "deep learning", "neural networks", "artificial intelligence") with biomedical applications
                ("biomedical", "medical", "clinical", "genomics", "drug discovery"). We included peer-reviewed articles published
                between January 2020 and December 2023 in English.
                """,
                'Results': """
                Our systematic search yielded 3,456 initial results, of which 1,247 met our inclusion criteria after screening.
                Deep learning approaches dominated the literature (n=815, 65.3%), with convolutional neural networks being most common
                for imaging applications and recurrent networks for sequential data analysis. Ensemble methods were the second most
                frequent approach (n=288, 23.1%), particularly for clinical prediction tasks. The most common application areas were
                medical imaging (34.2%), genomics and bioinformatics (28.7%), and drug discovery (19.3%).
                """,
                'Discussion': """
                Our findings demonstrate the maturation of machine learning in biomedical research, with increasingly sophisticated
                applications addressing complex clinical and biological questions. The dominance of deep learning reflects both
                the availability of large datasets and computational resources, as well as the success of these methods in achieving
                state-of-the-art performance. However, several challenges remain, including model interpretability, regulatory approval,
                and the need for robust validation in clinical settings.
                """,
                'Conclusion': """
                Machine learning has become an essential tool in biomedical research, with applications spanning from basic science
                to clinical practice. While technical advances continue to drive innovation, future success will depend on addressing
                challenges related to interpretability, validation, and clinical translation. The field is poised for continued growth
                as methods become more sophisticated and datasets continue to expand.
                """
            },
            'processing_mode': 'full',
            'has_fulltext': True
        },
        {
            'pmid': 'TEST_002',
            'title': 'CRISPR Gene Editing Safety Profile: A Meta-Analysis',
            'authors': ['Garcia, Rosa', 'Patel, Arjun K.'],
            'journal': 'Cell',
            'year': 2023,
            'abstract': """
            Background: CRISPR-Cas9 gene editing has shown therapeutic promise but safety concerns remain.
            Methods: Meta-analysis of 45 clinical trials involving CRISPR gene editing (2018-2023).
            Results: Off-target editing occurred in 2.3% of cases. Serious adverse events were rare (0.8%).
            Conclusions: CRISPR demonstrates acceptable safety profile for therapeutic applications.
            """,
            'processing_mode': 'abstracts',
            'has_fulltext': False
        },
        {
            'pmid': 'TEST_003',
            'title': 'COVID-19 Vaccination in Immunocompromised Patients: Real-World Evidence',
            'authors': ['Brown, Alice', 'Davis, Michael', 'Wilson, Sarah'],
            'journal': 'The Lancet',
            'year': 2023,
            'abstract': """
            Background: Immunocompromised patients may have reduced vaccine responses to COVID-19 vaccines.
            Methods: Retrospective cohort study of 25,847 immunocompromised patients across 15 medical centers.
            Results: Vaccine effectiveness was 68% vs 91% in healthy controls. Breakthrough infections were more common.
            Conclusions: Additional vaccine doses may be needed for immunocompromised populations.
            """,
            'sections': {
                'Background': """
                Immunocompromised patients, including those with cancer, autoimmune diseases, or organ transplants,
                represent a vulnerable population during the COVID-19 pandemic. These patients may have impaired immune
                responses to vaccination, potentially leaving them at higher risk for breakthrough infections.
                """,
                'Methods': """
                We conducted a retrospective cohort study using electronic health records from 15 major medical centers
                across the United States. The study included 25,847 immunocompromised patients and 50,000 healthy controls
                who received COVID-19 vaccination between December 2020 and August 2023.
                """,
                'Results': """
                Vaccine effectiveness against symptomatic COVID-19 was significantly lower in immunocompromised patients
                compared to healthy controls (68% vs 91%, p<0.001). Breakthrough infections occurred in 15.2% of
                immunocompromised patients vs 4.7% of controls. Hospitalization rates were also higher (3.2% vs 0.8%).
                """
            },
            'processing_mode': 'full',
            'has_fulltext': True
        }
    ]


@pytest.fixture
def edge_case_papers():
    """Edge case and malformed paper data for robustness testing"""
    return [
        {
            'pmid': 'EDGE_001',
            'title': '',  # Empty title
            'authors': [],
            'abstract': None,
            'sections': {},
            'processing_mode': 'full'
        },
        {
            'pmid': 'EDGE_002',
            'title': 'Very Short',
            'authors': ['Single, Author'],
            'abstract': 'Short.',
            'sections': {
                'Methods': 'Brief.',
                'Results': '',
                'Discussion': None
            },
            'processing_mode': 'full'
        },
        {
            'pmid': 'EDGE_003',
            'title': 'Extremely Long Title ' * 50,
            'authors': [f'Author{i}, Name{i}' for i in range(100)],
            'abstract': 'A' * 5000,
            'sections': {
                'Introduction': 'B' * 20000,
                'Methods': 'C' * 15000,
                'Results': 'D' * 25000
            },
            'processing_mode': 'full'
        },
        {
            'pmid': 'EDGE_004',
            'title': 'Unicode Test: 测试中文 العربية русский 🧬🔬',
            'authors': ['Müller, François', 'García-López, María José'],
            'abstract': 'Testing unicode: αβγ δεζ ηθι κλμ νξο πρσ τυφ χψω',
            'sections': {
                'Methods': 'Unicode methods: ∑∏∫∂∇ ≤≥≠±∞',
                'Results': 'Results with symbols: ℃℉°±×÷√'
            },
            'processing_mode': 'full'
        }
    ]


@pytest.fixture
def performance_test_data():
    """Large dataset for performance testing"""
    base_paper = {
        'pmid': 'PERF_BASE',
        'title': 'Performance Test Paper on Computational Biology Methods',
        'authors': ['Performance, Tester'],
        'journal': 'Test Journal',
        'year': 2023,
        'abstract': 'This is a performance test paper for benchmarking the RAG system capabilities.',
        'sections': {
            'Introduction': 'This section introduces the computational methods used in the study.',
            'Methods': 'We employed various computational algorithms for biological data analysis.',
            'Results': 'The results demonstrate significant computational efficiency improvements.',
            'Discussion': 'The implications of these findings for computational biology are significant.'
        },
        'processing_mode': 'full',
        'has_fulltext': True
    }

    # Generate multiple variations
    papers = []
    for i in range(100):
        paper = base_paper.copy()
        paper['pmid'] = f'PERF_{i:03d}'
        paper['title'] = f'Performance Test Paper {i}: {base_paper["title"]}'
        papers.append(paper)

    return papers


@pytest.fixture
def test_queries():
    """Standard test queries for RAG evaluation"""
    return [
        {
            'query': 'What machine learning methods were used in biomedical research?',
            'expected_sections': ['methods', 'introduction'],
            'query_type': 'methodological'
        },
        {
            'query': 'What were the main findings and results?',
            'expected_sections': ['results', 'conclusions'],
            'query_type': 'empirical'
        },
        {
            'query': 'What are the clinical implications and future directions?',
            'expected_sections': ['discussion', 'conclusions'],
            'query_type': 'synthesis'
        },
        {
            'query': 'How was data collection and analysis performed?',
            'expected_sections': ['methods'],
            'query_type': 'methodological'
        },
        {
            'query': 'What is the background and motivation for this research?',
            'expected_sections': ['introduction', 'background'],
            'query_type': 'conceptual'
        }
    ]


# Test markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and scalability tests"
    )
    config.addinivalue_line(
        "markers", "requires_chromadb: Tests that require ChromaDB installation"
    )
    config.addinivalue_line(
        "markers", "requires_models: Tests that require embedding models"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests that may take several minutes"
    )


# Skip conditions for optional dependencies
def pytest_collection_modifyitems(config, items):
    """Modify test items based on available dependencies"""
    try:
        import chromadb
        chromadb_available = True
    except ImportError:
        chromadb_available = False

    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False

    # Mark tests based on dependencies
    for item in items:
        if "requires_chromadb" in item.keywords and not chromadb_available:
            item.add_marker(pytest.mark.skip(reason="ChromaDB not available"))

        if "requires_models" in item.keywords and not torch_available:
            item.add_marker(pytest.mark.skip(reason="PyTorch not available"))


# Cleanup function
@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Automatically cleanup test artifacts after each test"""
    yield

    # Clean up temporary files and directories
    test_dirs = [
        'test_section_rag',
        'test_chromadb_data',
        'test_embeddings_cache'
    ]

    for test_dir in test_dirs:
        if Path(test_dir).exists():
            shutil.rmtree(test_dir, ignore_errors=True)