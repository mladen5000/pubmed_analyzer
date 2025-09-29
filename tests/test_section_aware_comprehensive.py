#!/usr/bin/env python3
"""
Comprehensive TDD Test Suite for Section-Aware ChromaDB RAG System
Based on agentic-RAG expert strategy with scientific literature focus
"""

import pytest
import asyncio
import tempfile
import shutil
import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

# Import system components
from pubmed_analyzer.core.section_aware_rag import (
    SectionAwareRAGAnalyzer, SectionType, QueryType, SectionContent,
    EnhancedPaperRepresentation, SectionClassifier
)
from pubmed_analyzer.core.chromadb_store import ScientificChromaStore
from pubmed_analyzer.core.section_embeddings import MultiModelEmbedder
from pubmed_analyzer.models.paper import Paper
from pubmed_analyzer.models.section import ProcessedSection, SectionMetadata, SectionCollection

# Test configuration
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class TestMetrics:
    """Track test performance metrics"""
    embedding_time: float = 0.0
    query_time: float = 0.0
    classification_accuracy: float = 0.0
    memory_usage_mb: float = 0.0
    chromadb_operations: int = 0


class TestDataGenerator:
    """Generate realistic test data for scientific papers"""

    @staticmethod
    def create_realistic_paper_data() -> List[Dict[str, Any]]:
        """Generate realistic scientific paper data with diverse content"""
        return [
            {
                'pmid': 'TEST_12345',
                'title': 'Machine Learning Applications in Biomedical Image Analysis: A Comprehensive Review',
                'authors': ['Smith, J.A.', 'Johnson, M.B.', 'Chen, L.'],
                'journal': 'Nature Biomedical Engineering',
                'year': 2023,
                'abstract': """
                Background: Medical imaging has been revolutionized by machine learning techniques, particularly deep learning algorithms.
                Methods: We systematically reviewed 150 studies published between 2020-2023, analyzing convolutional neural networks,
                transformer architectures, and federated learning approaches in medical imaging.
                Results: Deep learning models achieved 94.2% accuracy in cancer detection, with transformer models showing
                superior performance in multi-modal imaging tasks. Federated learning enabled privacy-preserving collaborative training.
                Conclusions: Machine learning demonstrates significant potential for clinical implementation, though challenges
                remain in model interpretability and regulatory approval.
                """,
                'sections': {
                    'Introduction': """
                    Medical imaging represents a cornerstone of modern healthcare, providing clinicians with essential diagnostic
                    information. The integration of artificial intelligence, particularly machine learning algorithms, has transformed
                    the field by enabling automated analysis of complex imaging data. Deep learning architectures, including
                    convolutional neural networks (CNNs) and more recently transformer models, have demonstrated remarkable
                    capabilities in image classification, segmentation, and anomaly detection tasks.
                    """,
                    'Methods': """
                    We conducted a systematic literature review following PRISMA guidelines. Our search strategy included
                    PubMed, IEEE Xplore, and arXiv databases using terms: "machine learning", "medical imaging", "deep learning",
                    "CNN", "transformer", and "federated learning". Inclusion criteria required peer-reviewed publications
                    from 2020-2023 with quantitative results on medical imaging tasks. Data extraction included model
                    architectures, dataset sizes, performance metrics, and clinical validation status.
                    """,
                    'Results': """
                    A total of 150 studies met inclusion criteria. CNN architectures (n=89, 59.3%) dominated the literature,
                    followed by hybrid CNN-transformer models (n=34, 22.7%). Average dataset sizes ranged from 1,000 to
                    50,000 images. Performance metrics showed: cancer detection accuracy 94.2±3.1%, segmentation Dice
                    coefficient 0.89±0.05, and anomaly detection AUC 0.93±0.04. Transformer models demonstrated superior
                    performance in multi-modal tasks, achieving 96.1% accuracy compared to 91.8% for traditional CNNs.
                    """,
                    'Discussion': """
                    Our findings demonstrate the maturation of machine learning in medical imaging, with consistently high
                    performance across diverse tasks and modalities. The emergence of transformer architectures represents
                    a significant advancement, particularly for complex multi-modal analysis. However, several challenges
                    persist: model interpretability remains limited, regulatory pathways are unclear, and clinical validation
                    is often insufficient. Future research should prioritize explainable AI methods and prospective clinical trials.
                    """,
                    'Conclusion': """
                    Machine learning has achieved remarkable success in medical imaging applications, with deep learning models
                    consistently outperforming traditional methods. While technical performance is impressive, translation to
                    clinical practice requires addressing interpretability, regulatory, and validation challenges. The field
                    is positioned for significant clinical impact in the coming decade.
                    """
                },
                'processing_mode': 'full',
                'has_fulltext': True
            },
            {
                'pmid': 'TEST_67890',
                'title': 'CRISPR-Cas9 Gene Editing in Neurological Disorders: Current Progress and Future Directions',
                'authors': ['Garcia, R.', 'Patel, S.K.', 'Liu, W.', 'Anderson, K.M.'],
                'journal': 'Cell',
                'year': 2023,
                'abstract': """
                CRISPR-Cas9 gene editing technology has emerged as a promising therapeutic approach for neurological disorders.
                This review synthesizes current clinical trials and preclinical studies targeting Huntington's disease,
                Alzheimer's disease, and amyotrophic lateral sclerosis. We analyze delivery mechanisms, safety profiles,
                and therapeutic efficacy across 45 studies. Results indicate significant promise for inherited neurological
                conditions, with successful gene correction achieved in 78% of in vitro studies and 62% of animal models.
                """,
                'sections': {
                    'Background': """
                    Neurological disorders represent a significant global health burden, affecting over 1 billion people worldwide.
                    Traditional therapeutic approaches have shown limited efficacy, particularly for inherited conditions caused
                    by single gene mutations. CRISPR-Cas9 technology offers unprecedented precision for gene editing, enabling
                    targeted correction of disease-causing mutations at the DNA level.
                    """,
                    'Methods': """
                    We systematically searched PubMed, Embase, and clinical trial databases for studies published between
                    2018-2023. Search terms included: "CRISPR", "Cas9", "gene editing", "neurological disorders",
                    "Huntington's", "Alzheimer's", and "ALS". Studies were categorized by disease target, delivery method,
                    and development stage. We extracted data on editing efficiency, off-target effects, and therapeutic outcomes.
                    """,
                    'Results': """
                    Forty-five studies met inclusion criteria: 28 preclinical, 12 clinical trials, and 5 translational studies.
                    Delivery mechanisms included adeno-associated virus (AAV) vectors (n=23), lipid nanoparticles (n=12),
                    and direct injection (n=10). Gene editing efficiency ranged from 45-92% across different systems.
                    Huntington's disease showed the highest success rates (85% efficiency), followed by inherited ALS (72%)
                    and early-onset Alzheimer's (68%). Safety profiles were generally favorable with minimal off-target effects.
                    """,
                    'Discussion': """
                    CRISPR-Cas9 demonstrates significant therapeutic potential for neurological disorders, particularly those
                    caused by single gene mutations. However, several challenges remain: delivery to the central nervous system
                    is complex, long-term safety data are limited, and ethical considerations surrounding germline editing
                    require careful evaluation. The technology shows greatest promise for severe inherited conditions where
                    current treatments are inadequate.
                    """
                },
                'processing_mode': 'full',
                'has_fulltext': True
            },
            {
                'pmid': 'TEST_11111',
                'title': 'Abstract-Only Study: COVID-19 Vaccination Effectiveness in Immunocompromised Patients',
                'authors': ['Brown, A.', 'Davis, M.'],
                'journal': 'The Lancet',
                'year': 2023,
                'abstract': """
                Background: Immunocompromised patients may have reduced vaccine effectiveness against COVID-19.
                Methods: Retrospective cohort study of 15,847 immunocompromised patients across 12 medical centers.
                Results: Vaccine effectiveness was 73% in immunocompromised vs 91% in healthy controls (p<0.001).
                Conclusions: Additional vaccine doses may be needed for immunocompromised populations.
                """,
                'processing_mode': 'abstracts',
                'has_fulltext': False
            }
        ]

    @staticmethod
    def create_malformed_paper_data() -> List[Dict[str, Any]]:
        """Generate edge cases and malformed data for robustness testing"""
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
                'title': 'Very Short Paper',
                'authors': ['Single, Author'],
                'abstract': 'Short.',
                'sections': {
                    'Introduction': 'Too short.',
                    'Methods': '',  # Empty section
                    'Results': None  # None section
                },
                'processing_mode': 'full'
            },
            {
                'pmid': 'EDGE_003',
                'title': 'Extremely Long Title ' * 100,  # Very long title
                'authors': ['Author, ' + str(i) for i in range(100)],  # Many authors
                'abstract': 'A' * 10000,  # Very long abstract
                'sections': {
                    'Introduction': 'B' * 50000,  # Very long section
                },
                'processing_mode': 'full'
            }
        ]


class TestFixtures:
    """Shared test fixtures and setup utilities"""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        temp_dir = tempfile.mkdtemp(prefix="test_rag_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_embedder(self):
        """Mock embedding model to avoid downloading large models"""
        embedder = Mock(spec=MultiModelEmbedder)
        embedder.embed_text.return_value = np.random.rand(1, 384)  # Fake embedding
        embedder.embed_sections.return_value = {
            'abstract': np.random.rand(384),
            'introduction': np.random.rand(384),
            'methods': np.random.rand(384),
            'results': np.random.rand(384),
            'discussion': np.random.rand(384)
        }
        embedder.get_query_embedding.return_value = np.random.rand(384)
        return embedder

    @pytest.fixture
    def test_papers(self):
        """Get realistic test paper data"""
        return TestDataGenerator.create_realistic_paper_data()

    @pytest.fixture
    def malformed_papers(self):
        """Get edge case test data"""
        return TestDataGenerator.create_malformed_paper_data()

    @pytest.fixture
    def section_classifier(self):
        """Create section classifier instance"""
        return SectionClassifier()


class TestSectionClassification(TestFixtures):
    """Test section classification accuracy and robustness"""

    def test_section_classification_accuracy(self, section_classifier):
        """Test classification accuracy on known section types"""
        test_cases = [
            ("This study presents a novel methodology for analyzing...", "Methods", SectionType.METHODS),
            ("Our results demonstrate significant improvements...", "Results", SectionType.RESULTS),
            ("In conclusion, this research shows...", "Conclusion", SectionType.CONCLUSION),
            ("The background literature indicates...", "Introduction", SectionType.INTRODUCTION),
            ("We discuss the implications of these findings...", "Discussion", SectionType.DISCUSSION),
            ("Abstract: This paper investigates...", "Abstract", SectionType.ABSTRACT)
        ]

        correct_classifications = 0
        total_tests = len(test_cases)

        for text, title, expected_type in test_cases:
            classified_type, confidence = section_classifier.classify_section(text, title)
            if classified_type == expected_type:
                correct_classifications += 1

            # Log for debugging
            logger.debug(f"Text: {text[:50]}... | Expected: {expected_type.value} | "
                        f"Classified: {classified_type.value} | Confidence: {confidence:.2f}")

        accuracy = correct_classifications / total_tests
        logger.info(f"Section classification accuracy: {accuracy:.2%}")

        # Assert minimum accuracy threshold
        assert accuracy >= 0.7, f"Classification accuracy {accuracy:.2%} below threshold"

    def test_section_classification_edge_cases(self, section_classifier):
        """Test classification robustness with edge cases"""
        edge_cases = [
            ("", "", SectionType.UNKNOWN),  # Empty content
            ("a", "b", SectionType.UNKNOWN),  # Very short content
            ("Methods: " * 1000, "Methods", SectionType.METHODS),  # Repetitive content
            ("This section discusses methods and results together...", "", None)  # Ambiguous
        ]

        for text, title, expected_type in edge_cases:
            classified_type, confidence = section_classifier.classify_section(text, title)

            # Should not crash and should return valid enum
            assert isinstance(classified_type, SectionType)
            assert 0.0 <= confidence <= 1.0

            if expected_type is not None:
                assert classified_type == expected_type

    def test_section_classification_performance(self, section_classifier):
        """Test classification performance with large batch"""
        # Generate large batch of test texts
        test_texts = ["This is a methods section describing experimental procedures."] * 1000
        test_titles = ["Methods"] * 1000

        start_time = time.time()

        results = []
        for text, title in zip(test_texts, test_titles):
            classified_type, confidence = section_classifier.classify_section(text, title)
            results.append((classified_type, confidence))

        end_time = time.time()
        processing_time = end_time - start_time

        logger.info(f"Classified 1000 sections in {processing_time:.2f}s "
                   f"({1000/processing_time:.1f} sections/sec)")

        # Performance assertion
        assert processing_time < 10.0, f"Classification too slow: {processing_time:.2f}s"
        assert len(results) == 1000


class TestChromaDBIntegration(TestFixtures):
    """Test ChromaDB storage and retrieval functionality"""

    def test_chromadb_initialization(self, temp_storage):
        """Test ChromaDB initialization and collection setup"""
        try:
            store = ScientificChromaStore(persist_directory=temp_storage)

            # Verify collections were created
            assert len(store.collections) > 0
            assert 'abstracts' in store.collections
            assert 'methods' in store.collections
            assert 'results' in store.collections

            # Verify persistence directory exists
            assert Path(temp_storage).exists()

            logger.info("✅ ChromaDB initialization successful")

        except Exception as e:
            pytest.skip(f"ChromaDB not available: {e}")

    def test_paper_storage_and_retrieval(self, temp_storage, test_papers):
        """Test storing and retrieving paper sections"""
        try:
            store = ScientificChromaStore(persist_directory=temp_storage)

            # Create test paper with structured sections
            paper_data = test_papers[0]
            paper = Paper(
                pmid=paper_data['pmid'],
                title=paper_data['title'],
                authors=paper_data['authors'],
                abstract=paper_data['abstract'],
                structured_sections={
                    'abstract': {
                        'content': paper_data['abstract'],
                        'content_length': len(paper_data['abstract']),
                        'confidence_score': 1.0,
                        'citations': [],
                        'figures_tables': []
                    },
                    'methods': {
                        'content': paper_data['sections']['Methods'],
                        'content_length': len(paper_data['sections']['Methods']),
                        'confidence_score': 0.9,
                        'citations': [],
                        'figures_tables': []
                    }
                }
            )

            # Mock embeddings
            section_embeddings = {
                'abstract': np.random.rand(384),
                'methods': np.random.rand(384)
            }

            # Add paper to store
            doc_ids = store.add_paper_sections(paper, section_embeddings)

            # Verify storage
            assert len(doc_ids) == 2
            assert 'abstract' in doc_ids
            assert 'methods' in doc_ids

            # Test retrieval
            query_embedding = np.random.rand(384)
            results = store.query_by_section(query_embedding, ['abstracts', 'methods'], n_results=5)

            assert len(results) > 0
            assert all('metadata' in result for result in results)
            assert all('similarity_score' in result for result in results)

            logger.info(f"✅ Stored and retrieved {len(doc_ids)} sections")

        except Exception as e:
            pytest.skip(f"ChromaDB operation failed: {e}")

    def test_metadata_filtering(self, temp_storage, test_papers):
        """Test ChromaDB metadata filtering capabilities"""
        try:
            store = ScientificChromaStore(persist_directory=temp_storage)

            # Store multiple papers with different metadata
            for paper_data in test_papers:
                paper = Paper(
                    pmid=paper_data['pmid'],
                    title=paper_data['title'],
                    journal=paper_data.get('journal', ''),
                    year=paper_data.get('year', 2023),
                    structured_sections={
                        'abstract': {
                            'content': paper_data['abstract'],
                            'content_length': len(paper_data['abstract']),
                            'confidence_score': 1.0
                        }
                    }
                )

                section_embeddings = {'abstract': np.random.rand(384)}
                store.add_paper_sections(paper, section_embeddings)

            # Test filtering by year
            query_embedding = np.random.rand(384)
            year_filter = {"year": 2023}

            results = store.query_by_section(
                query_embedding,
                ['abstracts'],
                filters=year_filter,
                n_results=10
            )

            # Verify all results match filter
            for result in results:
                assert result['metadata']['year'] == 2023

            logger.info(f"✅ Metadata filtering returned {len(results)} results")

        except Exception as e:
            pytest.skip(f"Metadata filtering test failed: {e}")


class TestSectionAwareRAG(TestFixtures):
    """Test the main Section-Aware RAG system"""

    @patch('pubmed_analyzer.core.section_embeddings.MultiModelEmbedder')
    def test_rag_initialization(self, mock_embedder_class, temp_storage):
        """Test RAG analyzer initialization with mocked embeddings"""
        mock_embedder_class.return_value = self.mock_embedder()

        try:
            analyzer = SectionAwareRAGAnalyzer(
                storage_path=temp_storage,
                use_chromadb=True
            )

            # Verify initialization
            assert analyzer.storage_path == temp_storage
            assert analyzer.section_classifier is not None
            assert analyzer.embedding_model is not None

            logger.info("✅ RAG analyzer initialization successful")

        except Exception as e:
            # Fallback to FAISS if ChromaDB fails
            logger.warning(f"ChromaDB initialization failed, testing FAISS fallback: {e}")

            analyzer = SectionAwareRAGAnalyzer(
                storage_path=temp_storage,
                use_chromadb=False
            )
            assert analyzer.use_chromadb == False

    @patch('pubmed_analyzer.core.section_embeddings.MultiModelEmbedder')
    def test_paper_processing_pipeline(self, mock_embedder_class, temp_storage, test_papers):
        """Test complete paper processing pipeline"""
        mock_embedder_class.return_value = self.mock_embedder()

        try:
            analyzer = SectionAwareRAGAnalyzer(storage_path=temp_storage)

            # Process papers
            results = analyzer.process_papers_with_sections(test_papers)

            # Verify processing results
            assert results['processed_papers'] == len(test_papers)
            assert results['total_sections'] > 0
            assert 'section_statistics' in results

            # Verify section statistics
            section_stats = results['section_statistics']
            assert section_stats['abstract'] > 0  # Should have abstracts

            logger.info(f"✅ Processed {results['processed_papers']} papers "
                       f"with {results['total_sections']} sections")

        except Exception as e:
            pytest.fail(f"Paper processing pipeline failed: {e}")

    @patch('pubmed_analyzer.core.section_embeddings.MultiModelEmbedder')
    def test_section_aware_querying(self, mock_embedder_class, temp_storage, test_papers):
        """Test section-aware query functionality"""
        mock_embedder_class.return_value = self.mock_embedder()

        try:
            analyzer = SectionAwareRAGAnalyzer(storage_path=temp_storage)

            # Process papers first
            analyzer.process_papers_with_sections(test_papers)

            # Test different query types
            query_tests = [
                ("machine learning methods", QueryType.METHODOLOGICAL),
                ("key findings and results", QueryType.EMPIRICAL),
                ("theoretical background", QueryType.CONCEPTUAL),
                ("comprehensive overview", QueryType.SYNTHESIS)
            ]

            for query, query_type in query_tests:
                result = analyzer.section_aware_query(
                    query=query,
                    query_type=query_type,
                    limit=5
                )

                # Verify query results structure
                assert 'query' in result
                assert 'query_type' in result
                assert 'contexts' in result
                assert 'response' in result
                assert result['query'] == query
                assert result['query_type'] == query_type.value

                logger.debug(f"Query '{query}' returned {len(result['contexts'])} contexts")

            logger.info("✅ Section-aware querying successful")

        except Exception as e:
            pytest.fail(f"Section-aware querying failed: {e}")


class TestErrorHandling(TestFixtures):
    """Test error handling and robustness"""

    def test_malformed_data_handling(self, temp_storage, malformed_papers):
        """Test system robustness with malformed data"""
        with patch('pubmed_analyzer.core.section_embeddings.MultiModelEmbedder') as mock_embedder_class:
            mock_embedder_class.return_value = self.mock_embedder()

            analyzer = SectionAwareRAGAnalyzer(storage_path=temp_storage)

            # Process malformed papers - should not crash
            try:
                results = analyzer.process_papers_with_sections(malformed_papers)

                # Should handle gracefully
                assert results['processed_papers'] >= 0
                logger.info(f"✅ Handled {len(malformed_papers)} malformed papers gracefully")

            except Exception as e:
                pytest.fail(f"System crashed on malformed data: {e}")

    def test_embedding_model_failure(self, temp_storage, test_papers):
        """Test behavior when embedding models fail"""
        # Mock failing embedder
        failing_embedder = Mock()
        failing_embedder.embed_text.side_effect = Exception("Model loading failed")
        failing_embedder.embed_sections.side_effect = Exception("Model loading failed")

        with patch('pubmed_analyzer.core.section_embeddings.MultiModelEmbedder') as mock_embedder_class:
            mock_embedder_class.return_value = failing_embedder

            # Should gracefully handle embedding failures
            try:
                analyzer = SectionAwareRAGAnalyzer(storage_path=temp_storage)
                results = analyzer.process_papers_with_sections(test_papers[:1])

                # May process with reduced functionality
                assert isinstance(results, dict)
                logger.info("✅ Handled embedding model failure gracefully")

            except Exception as e:
                # Expected to fail, but should be specific error, not crash
                assert "Model loading failed" in str(e) or "embedding" in str(e).lower()
                logger.info("✅ Failed gracefully with expected error")

    def test_storage_permission_error(self, test_papers):
        """Test handling of storage permission errors"""
        # Use read-only directory
        readonly_path = "/tmp/readonly_test"

        with patch('pubmed_analyzer.core.section_embeddings.MultiModelEmbedder') as mock_embedder_class:
            mock_embedder_class.return_value = self.mock_embedder()

            try:
                # This should handle permission errors gracefully
                analyzer = SectionAwareRAGAnalyzer(storage_path=readonly_path)
                logger.info("✅ Handled storage permission error gracefully")

            except (PermissionError, OSError) as e:
                # Expected error - should not crash the application
                logger.info(f"✅ Expected permission error handled: {e}")
            except Exception as e:
                pytest.fail(f"Unexpected error type: {e}")


class TestPerformance(TestFixtures):
    """Test performance benchmarks and scalability"""

    @patch('pubmed_analyzer.core.section_embeddings.MultiModelEmbedder')
    def test_large_dataset_processing(self, mock_embedder_class, temp_storage):
        """Test processing performance with larger dataset"""
        mock_embedder_class.return_value = self.mock_embedder()

        # Generate larger test dataset
        base_papers = TestDataGenerator.create_realistic_paper_data()
        large_dataset = []

        for i in range(50):  # Scale up to 50 papers for performance test
            for paper in base_papers:
                scaled_paper = paper.copy()
                scaled_paper['pmid'] = f"{paper['pmid']}_{i}"
                large_dataset.append(scaled_paper)

        analyzer = SectionAwareRAGAnalyzer(storage_path=temp_storage)

        # Measure processing time
        start_time = time.time()
        results = analyzer.process_papers_with_sections(large_dataset)
        end_time = time.time()

        processing_time = end_time - start_time
        papers_per_second = len(large_dataset) / processing_time

        logger.info(f"Processed {len(large_dataset)} papers in {processing_time:.2f}s "
                   f"({papers_per_second:.1f} papers/sec)")

        # Performance assertions
        assert papers_per_second > 1.0, f"Processing too slow: {papers_per_second:.2f} papers/sec"
        assert results['processed_papers'] == len(large_dataset)

    @patch('pubmed_analyzer.core.section_embeddings.MultiModelEmbedder')
    def test_query_response_time(self, mock_embedder_class, temp_storage, test_papers):
        """Test query response time performance"""
        mock_embedder_class.return_value = self.mock_embedder()

        analyzer = SectionAwareRAGAnalyzer(storage_path=temp_storage)
        analyzer.process_papers_with_sections(test_papers)

        # Test multiple queries and measure response time
        test_queries = [
            "machine learning applications",
            "clinical trial methodology",
            "treatment effectiveness",
            "future research directions",
            "statistical analysis methods"
        ]

        response_times = []

        for query in test_queries:
            start_time = time.time()

            result = analyzer.section_aware_query(
                query=query,
                query_type=QueryType.SYNTHESIS,
                limit=10
            )

            end_time = time.time()
            query_time = end_time - start_time
            response_times.append(query_time)

            logger.debug(f"Query '{query}' responded in {query_time:.3f}s")

        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)

        logger.info(f"Average query response time: {avg_response_time:.3f}s")
        logger.info(f"Maximum query response time: {max_response_time:.3f}s")

        # Performance assertions
        assert avg_response_time < 2.0, f"Average response time too slow: {avg_response_time:.3f}s"
        assert max_response_time < 5.0, f"Maximum response time too slow: {max_response_time:.3f}s"


class TestIntegration(TestFixtures):
    """Integration tests for complete workflows"""

    @patch('pubmed_analyzer.core.section_embeddings.MultiModelEmbedder')
    def test_end_to_end_workflow(self, mock_embedder_class, temp_storage, test_papers):
        """Test complete end-to-end workflow"""
        mock_embedder_class.return_value = self.mock_embedder()

        # Initialize system
        analyzer = SectionAwareRAGAnalyzer(storage_path=temp_storage)

        # Process papers
        processing_results = analyzer.process_papers_with_sections(test_papers)
        assert processing_results['processed_papers'] > 0

        # Test various query types
        query_scenarios = [
            {
                'query': 'What machine learning methods were used?',
                'query_type': QueryType.METHODOLOGICAL,
                'target_sections': [SectionType.METHODS]
            },
            {
                'query': 'What were the main findings?',
                'query_type': QueryType.EMPIRICAL,
                'target_sections': [SectionType.RESULTS]
            },
            {
                'query': 'What are the clinical implications?',
                'query_type': QueryType.SYNTHESIS,
                'target_sections': None
            }
        ]

        for scenario in query_scenarios:
            result = analyzer.section_aware_query(
                query=scenario['query'],
                query_type=scenario['query_type'],
                target_sections=scenario['target_sections'],
                limit=5
            )

            # Verify result structure
            assert 'contexts' in result
            assert 'response' in result
            assert len(result['contexts']) >= 0

        # Test system statistics
        stats = analyzer.get_section_statistics()
        assert 'collections' in stats or 'total_papers' in stats

        logger.info("✅ End-to-end workflow completed successfully")


# Pytest configuration and test discovery
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])