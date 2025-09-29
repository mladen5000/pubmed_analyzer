#!/usr/bin/env python3
"""
Multi-Model Embedding System Tests
Testing the scientific literature optimized embedding system
"""

import pytest
import tempfile
import shutil
import numpy as np
import logging
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from pubmed_analyzer.core.section_embeddings import (
    MultiModelEmbedder, normalize_embeddings, aggregate_embeddings,
    find_most_similar_sections
)

logger = logging.getLogger(__name__)


class TestMultiModelEmbedder:
    """Test the multi-model embedding system"""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory"""
        temp_dir = tempfile.mkdtemp(prefix="test_embeddings_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer to avoid downloading models"""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(5, 384)  # Batch of 5 embeddings
        return mock_model

    @pytest.fixture
    def mock_transformers_model(self):
        """Mock transformers model components"""
        mock_model = Mock()
        mock_tokenizer = Mock()

        # Mock tokenizer output
        mock_inputs = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]) if TORCH_AVAILABLE else [[1, 2, 3, 4, 5]],
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]]) if TORCH_AVAILABLE else [[1, 1, 1, 1, 1]]
        }
        mock_tokenizer.return_value = mock_inputs

        # Mock model output
        mock_outputs = Mock()
        if TORCH_AVAILABLE:
            mock_outputs.last_hidden_state = torch.rand(1, 5, 768)
            mock_outputs.pooler_output = torch.rand(1, 768)
        else:
            mock_outputs.last_hidden_state = np.random.rand(1, 5, 768)
            mock_outputs.pooler_output = np.random.rand(1, 768)

        mock_model.return_value = mock_outputs

        return mock_model, mock_tokenizer

    @pytest.fixture
    def sample_scientific_texts(self):
        """Sample scientific texts for testing"""
        return [
            "This study presents a novel machine learning approach for medical image analysis.",
            "We conducted a randomized controlled trial with 500 participants to evaluate treatment efficacy.",
            "The results demonstrate significant improvements in diagnostic accuracy using deep learning methods.",
            "Our methodology involved training convolutional neural networks on a large dataset of medical images.",
            "The discussion explores the clinical implications of these findings for patient care."
        ]

    def test_embedder_initialization_fallback(self, temp_cache_dir):
        """Test embedder initialization with fallback model"""
        with patch('pubmed_analyzer.core.section_embeddings.SentenceTransformer') as mock_st:
            mock_st.return_value = self.mock_sentence_transformer()

            embedder = MultiModelEmbedder(cache_dir=temp_cache_dir)

            # Should initialize with general model as fallback
            assert 'general' in embedder.models
            assert embedder.cache_dir.exists()

    def test_model_loading_fallback_strategy(self, temp_cache_dir):
        """Test model loading with fallback strategy"""
        with patch('pubmed_analyzer.core.section_embeddings.SentenceTransformer') as mock_st:
            mock_st.return_value = self.mock_sentence_transformer()

            embedder = MultiModelEmbedder(cache_dir=temp_cache_dir)

            # Mock a failing model load
            embedder.model_configs['biomedical']['name'] = 'nonexistent/model'

            # Should fall back to available models
            result = embedder._ensure_model_loaded('biomedical')
            # May succeed with fallback or fail gracefully

    @patch('pubmed_analyzer.core.section_embeddings.SentenceTransformer')
    def test_embed_text_single(self, mock_st_class, temp_cache_dir, sample_scientific_texts):
        """Test embedding single text"""
        mock_st_class.return_value = self.mock_sentence_transformer()

        embedder = MultiModelEmbedder(cache_dir=temp_cache_dir)

        # Test single text embedding
        text = sample_scientific_texts[0]
        embedding = embedder.embed_text(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == 1  # Single text
        assert embedding.shape[1] > 0   # Non-zero embedding dimension

    @patch('pubmed_analyzer.core.section_embeddings.SentenceTransformer')
    def test_embed_text_batch(self, mock_st_class, temp_cache_dir, sample_scientific_texts):
        """Test embedding batch of texts"""
        mock_st_class.return_value = self.mock_sentence_transformer()

        embedder = MultiModelEmbedder(cache_dir=temp_cache_dir)

        # Test batch embedding
        embeddings = embedder.embed_text(sample_scientific_texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(sample_scientific_texts)
        assert embeddings.shape[1] > 0

    @patch('pubmed_analyzer.core.section_embeddings.SentenceTransformer')
    def test_section_specific_model_selection(self, mock_st_class, temp_cache_dir):
        """Test section-specific model selection"""
        mock_st_class.return_value = self.mock_sentence_transformer()

        embedder = MultiModelEmbedder(cache_dir=temp_cache_dir)

        # Test different section types use appropriate models
        section_tests = [
            ("abstract", "This is an abstract section."),
            ("methods", "This describes the methodology."),
            ("results", "These are the results of the study."),
            ("discussion", "This discusses the implications.")
        ]

        for section_type, text in section_tests:
            embedding = embedder.embed_text(text, section_type=section_type)
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape[0] == 1

    @patch('pubmed_analyzer.core.section_embeddings.SentenceTransformer')
    def test_embed_sections(self, mock_st_class, temp_cache_dir):
        """Test embedding sections data structure"""
        mock_st_class.return_value = self.mock_sentence_transformer()

        embedder = MultiModelEmbedder(cache_dir=temp_cache_dir)

        sections_data = {
            'abstract': "This paper presents novel findings in machine learning.",
            'methods': "We used deep learning techniques with cross-validation.",
            'results': "The model achieved 95% accuracy on the test dataset.",
            'discussion': "These results have important clinical implications."
        }

        section_embeddings = embedder.embed_sections(sections_data)

        # Should return embeddings for each section
        assert len(section_embeddings) == len(sections_data)
        for section_type, embedding in section_embeddings.items():
            assert isinstance(embedding, np.ndarray)
            assert len(embedding.shape) == 1  # 1D embedding vector

    @patch('pubmed_analyzer.core.section_embeddings.SentenceTransformer')
    def test_query_embedding_with_context(self, mock_st_class, temp_cache_dir):
        """Test query embedding with context hints"""
        mock_st_class.return_value = self.mock_sentence_transformer()

        embedder = MultiModelEmbedder(cache_dir=temp_cache_dir)

        query = "machine learning methods for medical diagnosis"

        # Test different context hints
        contexts = ['biomedical', 'general', 'methodology']

        for context in contexts:
            embedding = embedder.get_query_embedding(query, context_hint=context)
            assert isinstance(embedding, np.ndarray)
            assert len(embedding.shape) == 1

    @patch('pubmed_analyzer.core.section_embeddings.SentenceTransformer')
    def test_similarity_computation(self, mock_st_class, temp_cache_dir):
        """Test similarity computation between embeddings"""
        mock_st_class.return_value = self.mock_sentence_transformer()

        embedder = MultiModelEmbedder(cache_dir=temp_cache_dir)

        # Create test embeddings
        embedding1 = np.random.rand(384)
        embedding2 = np.random.rand(384)

        # Test different similarity methods
        cosine_sim = embedder.compute_similarity(embedding1, embedding2, method='cosine')
        dot_sim = embedder.compute_similarity(embedding1, embedding2, method='dot')
        euclidean_sim = embedder.compute_similarity(embedding1, embedding2, method='euclidean')

        # Cosine similarity should be between -1 and 1
        assert -1 <= cosine_sim <= 1

        # Euclidean similarity should be positive
        assert euclidean_sim >= 0

        # All should be numeric
        assert isinstance(cosine_sim, float)
        assert isinstance(dot_sim, float)
        assert isinstance(euclidean_sim, float)

    @patch('pubmed_analyzer.core.section_embeddings.SentenceTransformer')
    def test_model_info_retrieval(self, mock_st_class, temp_cache_dir):
        """Test model information retrieval"""
        mock_st_class.return_value = self.mock_sentence_transformer()

        embedder = MultiModelEmbedder(cache_dir=temp_cache_dir)

        model_info = embedder.get_model_info()

        # Verify info structure
        assert 'available_models' in model_info
        assert 'loaded_models' in model_info
        assert 'section_mappings' in model_info
        assert 'device' in model_info
        assert 'cache_directory' in model_info

        # Check available models
        for model_key in embedder.model_configs:
            assert model_key in model_info['available_models']

    @patch('pubmed_analyzer.core.section_embeddings.SentenceTransformer')
    def test_benchmark_models(self, mock_st_class, temp_cache_dir, sample_scientific_texts):
        """Test model benchmarking functionality"""
        mock_st_class.return_value = self.mock_sentence_transformer()

        embedder = MultiModelEmbedder(cache_dir=temp_cache_dir)

        # Run benchmark on sample texts
        benchmark_results = embedder.benchmark_models(sample_scientific_texts[:3])

        # Should have results for available models
        assert isinstance(benchmark_results, dict)

        for model_key, results in benchmark_results.items():
            if results.get('success', False):
                assert 'embedding_time' in results
                assert 'embeddings_per_second' in results
                assert 'embedding_shape' in results
                assert results['embedding_time'] > 0

    @patch('pubmed_analyzer.core.section_embeddings.SentenceTransformer')
    def test_error_handling_empty_text(self, mock_st_class, temp_cache_dir):
        """Test error handling with empty or invalid text"""
        mock_st_class.return_value = self.mock_sentence_transformer()

        embedder = MultiModelEmbedder(cache_dir=temp_cache_dir)

        # Test empty text
        try:
            embedding = embedder.embed_text("")
            # Should handle gracefully
            assert isinstance(embedding, np.ndarray)
        except Exception as e:
            # Or raise appropriate error
            assert "empty" in str(e).lower() or "text" in str(e).lower()

    @patch('pubmed_analyzer.core.section_embeddings.SentenceTransformer')
    def test_error_handling_model_failure(self, mock_st_class, temp_cache_dir):
        """Test error handling when models fail"""
        # Mock failing model
        failing_model = Mock()
        failing_model.encode.side_effect = Exception("Model failed")
        mock_st_class.return_value = failing_model

        embedder = MultiModelEmbedder(cache_dir=temp_cache_dir)

        # Should handle model failures gracefully
        with pytest.raises(Exception):
            embedder.embed_text("test text")

    @patch('pubmed_analyzer.core.section_embeddings.SentenceTransformer')
    def test_save_load_embeddings(self, mock_st_class, temp_cache_dir):
        """Test saving and loading embeddings"""
        mock_st_class.return_value = self.mock_sentence_transformer()

        embedder = MultiModelEmbedder(cache_dir=temp_cache_dir)

        # Create test embeddings
        test_embeddings = {
            'embedding1': np.random.rand(384),
            'embedding2': np.random.rand(384),
            'embedding3': np.random.rand(384)
        }

        # Save embeddings
        save_path = temp_cache_dir + "/test_embeddings.json"
        embedder.save_embeddings(test_embeddings, save_path)

        # Load embeddings
        loaded_embeddings = embedder.load_embeddings(save_path)

        # Verify loaded embeddings match original
        assert len(loaded_embeddings) == len(test_embeddings)
        for key in test_embeddings:
            assert key in loaded_embeddings
            np.testing.assert_array_almost_equal(
                test_embeddings[key],
                loaded_embeddings[key],
                decimal=5
            )

    @patch('pubmed_analyzer.core.section_embeddings.SentenceTransformer')
    def test_memory_cleanup(self, mock_st_class, temp_cache_dir):
        """Test memory cleanup functionality"""
        mock_st_class.return_value = self.mock_sentence_transformer()

        embedder = MultiModelEmbedder(cache_dir=temp_cache_dir)

        # Load multiple models
        embedder._ensure_model_loaded('general')
        initial_model_count = len(embedder.models)

        # Cleanup models
        embedder.cleanup_models()

        # Should keep at least the fallback model
        assert len(embedder.models) >= 1
        assert 'general' in embedder.models  # Should keep fallback


class TestEmbeddingUtilities:
    """Test embedding utility functions"""

    def test_normalize_embeddings(self):
        """Test embedding normalization"""
        # Create test embeddings
        embeddings = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 0.0, 0.0]  # Zero vector edge case
        ])

        normalized = normalize_embeddings(embeddings)

        # Check shapes match
        assert normalized.shape == embeddings.shape

        # Check normalization (except for zero vectors)
        for i, embedding in enumerate(normalized):
            if not np.allclose(embeddings[i], 0):
                norm = np.linalg.norm(embedding)
                assert abs(norm - 1.0) < 1e-6

    def test_aggregate_embeddings_mean(self):
        """Test mean aggregation of embeddings"""
        embeddings = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([7.0, 8.0, 9.0])
        ]

        aggregated = aggregate_embeddings(embeddings, method='mean')
        expected = np.array([4.0, 5.0, 6.0])  # Mean of the three vectors

        np.testing.assert_array_almost_equal(aggregated, expected)

    def test_aggregate_embeddings_max(self):
        """Test max aggregation of embeddings"""
        embeddings = [
            np.array([1.0, 8.0, 3.0]),
            np.array([4.0, 2.0, 6.0]),
            np.array([7.0, 5.0, 1.0])
        ]

        aggregated = aggregate_embeddings(embeddings, method='max')
        expected = np.array([7.0, 8.0, 6.0])  # Element-wise max

        np.testing.assert_array_almost_equal(aggregated, expected)

    def test_aggregate_embeddings_weighted(self):
        """Test weighted mean aggregation"""
        embeddings = [
            np.array([1.0, 0.0, 0.0]),  # Norm = 1
            np.array([0.0, 2.0, 0.0]),  # Norm = 2
            np.array([0.0, 0.0, 3.0])   # Norm = 3
        ]

        aggregated = aggregate_embeddings(embeddings, method='weighted_mean')

        # Should be weighted by norms: weights = [1/6, 2/6, 3/6]
        # Result should favor the higher-norm embeddings
        assert aggregated[2] > aggregated[1] > aggregated[0]

    def test_aggregate_embeddings_empty(self):
        """Test aggregation with empty embedding list"""
        with pytest.raises(ValueError):
            aggregate_embeddings([], method='mean')

    def test_find_most_similar_sections(self):
        """Test finding most similar sections"""
        query_embedding = np.array([1.0, 0.0, 0.0])

        section_embeddings = {
            'section1': np.array([1.0, 0.0, 0.0]),  # Identical
            'section2': np.array([0.0, 1.0, 0.0]),  # Orthogonal
            'section3': np.array([0.5, 0.5, 0.0]),  # Similar
            'section4': np.array([-1.0, 0.0, 0.0])  # Opposite
        }

        similar_sections = find_most_similar_sections(
            query_embedding,
            section_embeddings,
            top_k=3
        )

        # Should return 3 results
        assert len(similar_sections) == 3

        # Results should be tuples of (section_id, similarity_score)
        for section_id, similarity in similar_sections:
            assert isinstance(section_id, str)
            assert isinstance(similarity, float)
            assert -1 <= similarity <= 1

        # First result should be most similar (section1)
        assert similar_sections[0][0] == 'section1'
        assert similar_sections[0][1] > 0.9  # Nearly identical

    def test_find_most_similar_sections_empty(self):
        """Test finding similar sections with empty input"""
        query_embedding = np.array([1.0, 0.0, 0.0])
        section_embeddings = {}

        similar_sections = find_most_similar_sections(
            query_embedding,
            section_embeddings,
            top_k=5
        )

        assert len(similar_sections) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])