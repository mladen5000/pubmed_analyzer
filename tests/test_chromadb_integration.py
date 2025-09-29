#!/usr/bin/env python3
"""
ChromaDB Integration Tests
Focused testing for the ScientificChromaStore component
"""

import pytest
import tempfile
import shutil
import numpy as np
import logging
from typing import Dict, List, Any
from pathlib import Path

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from pubmed_analyzer.core.chromadb_store import ScientificChromaStore
from pubmed_analyzer.models.paper import Paper

logger = logging.getLogger(__name__)


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not available")
class TestChromaDBStore:
    """Test ChromaDB storage operations"""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        temp_dir = tempfile.mkdtemp(prefix="test_chromadb_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def chromadb_store(self, temp_storage):
        """Create ChromaDB store instance"""
        return ScientificChromaStore(persist_directory=temp_storage)

    @pytest.fixture
    def sample_paper(self):
        """Create sample paper with structured sections"""
        return Paper(
            pmid="TEST_PAPER_001",
            title="Sample Research Paper on Machine Learning",
            authors=["Smith, J.", "Doe, A."],
            journal="Nature",
            year=2023,
            abstract="This paper presents novel machine learning techniques for medical diagnosis.",
            structured_sections={
                'abstract': {
                    'content': "This paper presents novel machine learning techniques for medical diagnosis. We developed a deep learning model that achieves 95% accuracy on medical imaging tasks.",
                    'content_length': 140,
                    'confidence_score': 1.0,
                    'citations': [],
                    'figures_tables': []
                },
                'methods': {
                    'content': "We used a convolutional neural network with ResNet architecture. The model was trained on 10,000 medical images using Adam optimizer with learning rate 0.001.",
                    'content_length': 160,
                    'confidence_score': 0.95,
                    'citations': ['ref1', 'ref2'],
                    'figures_tables': ['Figure 1']
                },
                'results': {
                    'content': "Our model achieved 95.2% accuracy on the test set, outperforming baseline methods by 8.3%. Sensitivity was 94.1% and specificity was 96.8%.",
                    'content_length': 140,
                    'confidence_score': 0.9,
                    'citations': [],
                    'figures_tables': ['Table 1', 'Figure 2']
                }
            }
        )

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing"""
        return {
            'abstract': np.random.rand(384),
            'methods': np.random.rand(384),
            'results': np.random.rand(384)
        }

    def test_store_initialization(self, chromadb_store, temp_storage):
        """Test ChromaDB store initialization"""
        # Verify store was created
        assert chromadb_store.persist_directory == Path(temp_storage)
        assert chromadb_store.client is not None

        # Verify collections were created
        expected_collections = ['abstracts', 'methods', 'results', 'discussions', 'conclusions']
        for collection_name in expected_collections:
            assert collection_name in chromadb_store.collections

        # Verify persistence directory exists
        assert Path(temp_storage).exists()

    def test_add_paper_sections(self, chromadb_store, sample_paper, sample_embeddings):
        """Test adding paper sections to ChromaDB"""
        # Add paper sections
        doc_ids = chromadb_store.add_paper_sections(sample_paper, sample_embeddings)

        # Verify document IDs were returned
        assert len(doc_ids) == 3
        assert 'abstract' in doc_ids
        assert 'methods' in doc_ids
        assert 'results' in doc_ids

        # Verify all IDs are strings and not empty
        for section_type, doc_id in doc_ids.items():
            assert isinstance(doc_id, str)
            assert len(doc_id) > 0
            assert sample_paper.clean_pmid in doc_id

    def test_query_by_section(self, chromadb_store, sample_paper, sample_embeddings):
        """Test querying sections by type"""
        # Add paper first
        chromadb_store.add_paper_sections(sample_paper, sample_embeddings)

        # Create query embedding
        query_embedding = np.random.rand(384)

        # Test single section type query
        results = chromadb_store.query_by_section(
            query_embedding=query_embedding,
            section_types=['abstracts'],
            n_results=5
        )

        assert len(results) >= 1
        for result in results:
            assert 'content' in result
            assert 'metadata' in result
            assert 'similarity_score' in result
            assert 'section_type' in result
            assert result['section_type'] == 'abstracts'

    def test_query_multiple_sections(self, chromadb_store, sample_paper, sample_embeddings):
        """Test querying multiple section types"""
        # Add paper first
        chromadb_store.add_paper_sections(sample_paper, sample_embeddings)

        # Query multiple sections
        query_embedding = np.random.rand(384)
        results = chromadb_store.query_by_section(
            query_embedding=query_embedding,
            section_types=['abstracts', 'methods', 'results'],
            n_results=10
        )

        # Should get results from different section types
        section_types_found = set(result['section_type'] for result in results)
        assert len(section_types_found) > 1

    def test_metadata_filtering(self, chromadb_store, sample_paper, sample_embeddings):
        """Test metadata filtering in queries"""
        # Add paper first
        chromadb_store.add_paper_sections(sample_paper, sample_embeddings)

        # Query with metadata filter
        query_embedding = np.random.rand(384)
        filters = {"pmid": sample_paper.clean_pmid}

        results = chromadb_store.query_by_section(
            query_embedding=query_embedding,
            section_types=['abstracts'],
            filters=filters,
            n_results=5
        )

        # All results should match the filter
        for result in results:
            assert result['metadata']['pmid'] == sample_paper.clean_pmid

    def test_cross_section_analysis(self, chromadb_store, sample_paper, sample_embeddings):
        """Test cross-section analysis functionality"""
        # Add paper first
        chromadb_store.add_paper_sections(sample_paper, sample_embeddings)

        # Perform cross-section analysis
        query_embedding = np.random.rand(384)
        results = chromadb_store.query_cross_sections(
            query_embedding=query_embedding,
            primary_sections=['methods'],
            secondary_sections=['results'],
            n_results=3
        )

        # Verify result structure
        assert 'primary' in results
        assert 'secondary' in results
        assert 'cross_section_papers' in results
        assert 'cross_section_count' in results

        # Should find cross-section papers
        assert results['cross_section_count'] >= 1
        assert sample_paper.clean_pmid in results['cross_section_papers']

    def test_research_context_filtering(self, chromadb_store, sample_paper, sample_embeddings):
        """Test research context-based filtering"""
        # Add paper first
        chromadb_store.add_paper_sections(sample_paper, sample_embeddings)

        # Test different research contexts
        contexts = ['methodology', 'findings', 'background']
        query_embedding = np.random.rand(384)

        for context in contexts:
            results = chromadb_store.filter_by_research_context(
                query_embedding=query_embedding,
                research_context=context,
                n_results=5
            )

            # Should return results (may be empty for some contexts)
            assert isinstance(results, list)
            for result in results:
                assert 'content' in result
                assert 'metadata' in result

    def test_get_paper_sections(self, chromadb_store, sample_paper, sample_embeddings):
        """Test retrieving all sections for a specific paper"""
        # Add paper first
        chromadb_store.add_paper_sections(sample_paper, sample_embeddings)

        # Retrieve all sections for the paper
        paper_sections = chromadb_store.get_paper_sections(sample_paper.clean_pmid)

        # Should find sections for this paper
        assert len(paper_sections) > 0

        # Verify section structure
        for section_type, section_data in paper_sections.items():
            assert 'content' in section_data
            assert 'metadata' in section_data

    def test_collection_statistics(self, chromadb_store, sample_paper, sample_embeddings):
        """Test collection statistics functionality"""
        # Get initial stats
        initial_stats = chromadb_store.get_collection_stats()
        assert 'total_collections' in initial_stats
        assert 'collection_details' in initial_stats
        assert 'total_documents' in initial_stats

        initial_document_count = initial_stats['total_documents']

        # Add paper
        chromadb_store.add_paper_sections(sample_paper, sample_embeddings)

        # Get updated stats
        updated_stats = chromadb_store.get_collection_stats()
        assert updated_stats['total_documents'] > initial_document_count

    def test_delete_paper(self, chromadb_store, sample_paper, sample_embeddings):
        """Test deleting a paper from ChromaDB"""
        # Add paper first
        doc_ids = chromadb_store.add_paper_sections(sample_paper, sample_embeddings)
        assert len(doc_ids) > 0

        # Delete the paper
        deletion_status = chromadb_store.delete_paper(sample_paper.clean_pmid)

        # Verify deletion status
        assert isinstance(deletion_status, dict)
        for section_type, status in deletion_status.items():
            assert isinstance(status, bool)

        # Verify paper is no longer retrievable
        paper_sections = chromadb_store.get_paper_sections(sample_paper.clean_pmid)
        assert len(paper_sections) == 0

    def test_search_by_methodology(self, chromadb_store):
        """Test methodology-specific search"""
        # Create paper with methodology metadata
        paper = Paper(
            pmid="METHODOLOGY_TEST",
            structured_sections={
                'methods': {
                    'content': "We used deep learning and statistical analysis methods.",
                    'content_length': 55,
                    'confidence_score': 0.9
                }
            }
        )

        # Set methodology in metadata
        paper.structured_sections['methods']['research_methodology'] = 'machine_learning'

        embeddings = {'methods': np.random.rand(384)}
        chromadb_store.add_paper_sections(paper, embeddings)

        # Search by methodology
        query_embedding = np.random.rand(384)
        results = chromadb_store.search_by_methodology(
            query_embedding=query_embedding,
            methodology_types=['machine_learning'],
            n_results=5
        )

        # Should find the paper
        assert len(results) > 0

    def test_search_significant_findings(self, chromadb_store):
        """Test searching for papers with significant findings"""
        # Create paper with significant findings
        paper = Paper(
            pmid="FINDINGS_TEST",
            structured_sections={
                'results': {
                    'content': "We found significant improvements in accuracy (p<0.001).",
                    'content_length': 60,
                    'confidence_score': 0.95,
                    'finding_type': 'significant',
                    'findings_count': 3
                }
            }
        )

        embeddings = {'results': np.random.rand(384)}
        chromadb_store.add_paper_sections(paper, embeddings)

        # Search for significant findings
        query_embedding = np.random.rand(384)
        results = chromadb_store.search_significant_findings(
            query_embedding=query_embedding,
            min_findings_count=1,
            n_results=5
        )

        # Should find the paper
        assert len(results) > 0

    def test_persistence_across_sessions(self, temp_storage, sample_paper, sample_embeddings):
        """Test that data persists across different store instances"""
        # Create first store instance and add data
        store1 = ScientificChromaStore(persist_directory=temp_storage)
        doc_ids = store1.add_paper_sections(sample_paper, sample_embeddings)
        assert len(doc_ids) > 0

        # Create second store instance (simulating restart)
        store2 = ScientificChromaStore(persist_directory=temp_storage)

        # Verify data persists
        paper_sections = store2.get_paper_sections(sample_paper.clean_pmid)
        assert len(paper_sections) > 0

    def test_error_handling_invalid_paper(self, chromadb_store):
        """Test error handling with invalid paper data"""
        # Paper without structured sections
        invalid_paper = Paper(pmid="INVALID_001")
        embeddings = {}

        # Should handle gracefully
        doc_ids = chromadb_store.add_paper_sections(invalid_paper, embeddings)
        assert len(doc_ids) == 0  # No sections added

    def test_error_handling_mismatched_embeddings(self, chromadb_store, sample_paper):
        """Test error handling with mismatched embeddings"""
        # Embeddings that don't match sections
        mismatched_embeddings = {
            'nonexistent_section': np.random.rand(384)
        }

        # Should handle gracefully without crashing
        doc_ids = chromadb_store.add_paper_sections(sample_paper, mismatched_embeddings)
        # May return empty dict or partial results, but shouldn't crash

    def test_large_content_handling(self, chromadb_store):
        """Test handling of very large content"""
        # Create paper with very large sections
        large_content = "A" * 100000  # Very large section
        large_paper = Paper(
            pmid="LARGE_TEST",
            structured_sections={
                'abstract': {
                    'content': large_content,
                    'content_length': len(large_content),
                    'confidence_score': 1.0
                }
            }
        )

        embeddings = {'abstract': np.random.rand(384)}

        # Should handle large content without issues
        doc_ids = chromadb_store.add_paper_sections(large_paper, embeddings)
        assert 'abstract' in doc_ids

        # Should be able to query it
        query_embedding = np.random.rand(384)
        results = chromadb_store.query_by_section(
            query_embedding=query_embedding,
            section_types=['abstracts'],
            n_results=1
        )
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])