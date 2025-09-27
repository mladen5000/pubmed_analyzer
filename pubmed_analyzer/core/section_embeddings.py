#!/usr/bin/env python3
"""
Multi-Model Embedding System for Section-Aware RAG
Scientific literature-optimized embeddings with section-specific models
"""

import logging
import warnings
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import json

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

from ..utils.section_chunker import SectionChunk

logger = logging.getLogger(__name__)

# Suppress tokenizer warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


class MultiModelEmbedder:
    """
    Multi-model embedding system optimized for scientific literature
    with section-specific model selection and fallback strategies
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize multi-model embedder with scientific models

        Args:
            cache_dir: Directory to cache models (optional)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "pubmed_analyzer_models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Model configurations for different scientific domains
        self.model_configs = {
            'scientific': {
                'name': 'allenai/scibert_scivocab_uncased',
                'type': 'transformers',
                'description': 'SciBERT - General scientific literature',
                'max_seq_length': 512,
                'embedding_dim': 768
            },
            'biomedical': {
                'name': 'dmis-lab/biobert-base-cased-v1.1',
                'type': 'transformers',
                'description': 'BioBERT - Biomedical literature',
                'max_seq_length': 512,
                'embedding_dim': 768
            },
            'general': {
                'name': 'sentence-transformers/all-MiniLM-L6-v2',
                'type': 'sentence_transformers',
                'description': 'General purpose sentence transformer (fallback)',
                'max_seq_length': 256,
                'embedding_dim': 384
            },
            'scientific_sentence': {
                'name': 'sentence-transformers/allenai-specter',
                'type': 'sentence_transformers',
                'description': 'SPECTER - Scientific paper embeddings',
                'max_seq_length': 512,
                'embedding_dim': 768
            }
        }

        # Section-specific model mapping
        self.section_model_mapping = {
            'abstract': 'biomedical',      # Abstracts often contain biomedical terms
            'introduction': 'scientific',   # General scientific context
            'methods': 'scientific',       # Methodological descriptions
            'results': 'biomedical',       # Often contains biomedical findings
            'discussion': 'scientific',    # Analysis and interpretation
            'conclusion': 'scientific',    # General scientific conclusions
            'references': 'general'        # Citations are less domain-specific
        }

        # Model instances (lazy loaded)
        self.models = {}
        self.tokenizers = {}

        # Device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔧 Using device: {self.device}")

        # Initialize with general model first (most likely to succeed)
        self._initialize_fallback_model()

    def _initialize_fallback_model(self):
        """Initialize the fallback general model"""
        try:
            logger.info("🔄 Loading fallback general model...")
            self.models['general'] = SentenceTransformer(
                self.model_configs['general']['name'],
                cache_folder=str(self.cache_dir)
            )
            logger.info("✅ General model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load fallback model: {e}")
            raise RuntimeError("Cannot initialize any embedding model")

    def _load_transformers_model(self, model_key: str) -> Tuple[AutoModel, AutoTokenizer]:
        """Load a transformers-based model"""
        config = self.model_configs[model_key]
        model_name = config['name']

        try:
            logger.info(f"🔄 Loading {config['description']} ({model_name})...")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir),
                trust_remote_code=True
            )

            # Load model
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir),
                trust_remote_code=True
            )
            model.to(self.device)
            model.eval()

            logger.info(f"✅ {config['description']} loaded successfully")
            return model, tokenizer

        except Exception as e:
            logger.error(f"❌ Failed to load {model_name}: {e}")
            raise

    def _load_sentence_transformer_model(self, model_key: str) -> SentenceTransformer:
        """Load a sentence-transformers model"""
        config = self.model_configs[model_key]
        model_name = config['name']

        try:
            logger.info(f"🔄 Loading {config['description']} ({model_name})...")

            model = SentenceTransformer(
                model_name,
                cache_folder=str(self.cache_dir),
                device=str(self.device)
            )

            logger.info(f"✅ {config['description']} loaded successfully")
            return model

        except Exception as e:
            logger.error(f"❌ Failed to load {model_name}: {e}")
            raise

    def _ensure_model_loaded(self, model_key: str) -> bool:
        """Ensure a specific model is loaded, with fallback handling"""
        if model_key in self.models:
            return True

        config = self.model_configs.get(model_key)
        if not config:
            logger.warning(f"⚠️ Unknown model key: {model_key}")
            return False

        try:
            if config['type'] == 'transformers':
                model, tokenizer = self._load_transformers_model(model_key)
                self.models[model_key] = model
                self.tokenizers[model_key] = tokenizer
            elif config['type'] == 'sentence_transformers':
                model = self._load_sentence_transformer_model(model_key)
                self.models[model_key] = model
            else:
                logger.error(f"❌ Unknown model type: {config['type']}")
                return False

            return True

        except Exception as e:
            logger.warning(f"⚠️ Failed to load {model_key}: {e}")
            return False

    def _get_transformer_embeddings(self, texts: List[str], model_key: str) -> np.ndarray:
        """Get embeddings from transformers-based models"""
        model = self.models[model_key]
        tokenizer = self.tokenizers[model_key]
        config = self.model_configs[model_key]

        embeddings = []

        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=config['max_seq_length'],
                    truncation=True,
                    padding=True
                ).to(self.device)

                # Get model outputs
                outputs = model(**inputs)

                # Use CLS token embedding for sentence-level representation
                # or mean pooling of last hidden states
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embedding = outputs.pooler_output
                else:
                    # Mean pooling
                    token_embeddings = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask']
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

                # Normalize embedding
                embedding = F.normalize(embedding, p=2, dim=1)
                embeddings.append(embedding.cpu().numpy())

        return np.vstack(embeddings)

    def _get_sentence_transformer_embeddings(self, texts: List[str], model_key: str) -> np.ndarray:
        """Get embeddings from sentence-transformers models"""
        model = self.models[model_key]
        return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def embed_text(self,
                   texts: Union[str, List[str]],
                   model_key: Optional[str] = None,
                   section_type: Optional[str] = None) -> np.ndarray:
        """
        Generate embeddings for text(s) using appropriate model

        Args:
            texts: Text or list of texts to embed
            model_key: Specific model to use (optional)
            section_type: Section type for automatic model selection (optional)

        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        # Determine best model to use
        if model_key is None:
            if section_type and section_type in self.section_model_mapping:
                model_key = self.section_model_mapping[section_type]
            else:
                model_key = 'scientific'  # Default to scientific model

        # Try primary model, fallback to general if needed
        primary_model = model_key
        fallback_models = ['general', 'scientific', 'biomedical']

        for attempt_model in [primary_model] + [m for m in fallback_models if m != primary_model]:
            if self._ensure_model_loaded(attempt_model):
                try:
                    config = self.model_configs[attempt_model]

                    if config['type'] == 'transformers':
                        embeddings = self._get_transformer_embeddings(texts, attempt_model)
                    else:
                        embeddings = self._get_sentence_transformer_embeddings(texts, attempt_model)

                    if attempt_model != primary_model:
                        logger.warning(f"⚠️ Used fallback model {attempt_model} instead of {primary_model}")

                    return embeddings

                except Exception as e:
                    logger.warning(f"⚠️ Model {attempt_model} failed during embedding: {e}")
                    continue

        raise RuntimeError("All embedding models failed")

    def embed_section_chunks(self, chunks: List[SectionChunk]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for section chunks using optimal models

        Args:
            chunks: List of section chunks

        Returns:
            Dict mapping chunk IDs to embeddings
        """
        chunk_embeddings = {}

        # Group chunks by section type for batch processing
        chunks_by_section = {}
        for chunk in chunks:
            section_type = chunk.metadata.section_type
            if section_type not in chunks_by_section:
                chunks_by_section[section_type] = []
            chunks_by_section[section_type].append(chunk)

        # Process each section type with its optimal model
        for section_type, section_chunks in chunks_by_section.items():
            try:
                # Extract text content
                texts = [chunk.content for chunk in section_chunks]

                # Generate embeddings
                embeddings = self.embed_text(texts, section_type=section_type)

                # Map embeddings to chunk IDs
                for chunk, embedding in zip(section_chunks, embeddings):
                    chunk_embeddings[chunk.metadata.chunk_id] = embedding

                logger.debug(f"✅ Generated embeddings for {len(section_chunks)} {section_type} chunks")

            except Exception as e:
                logger.error(f"❌ Failed to embed {section_type} chunks: {e}")
                continue

        return chunk_embeddings

    def embed_sections(self, sections_data: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for paper sections

        Args:
            sections_data: Dict mapping section types to content

        Returns:
            Dict mapping section types to embeddings
        """
        section_embeddings = {}

        for section_type, content in sections_data.items():
            if not content.strip():
                continue

            try:
                embedding = self.embed_text(content, section_type=section_type)
                section_embeddings[section_type] = embedding[0]  # Single text returns 1D array

                logger.debug(f"✅ Generated embedding for {section_type} section")

            except Exception as e:
                logger.error(f"❌ Failed to embed {section_type} section: {e}")
                continue

        return section_embeddings

    def get_query_embedding(self, query: str, context_hint: Optional[str] = None) -> np.ndarray:
        """
        Generate embedding for search query with context-aware model selection

        Args:
            query: Search query text
            context_hint: Hint about query context (methodology, findings, etc.)

        Returns:
            Query embedding vector
        """
        # Select model based on context hint
        model_key = 'scientific'  # Default

        if context_hint:
            context_hint = context_hint.lower()
            if any(bio_term in context_hint for bio_term in ['biomedical', 'medical', 'clinical', 'biological']):
                model_key = 'biomedical'
            elif any(gen_term in context_hint for gen_term in ['general', 'overview', 'summary']):
                model_key = 'general'

        embedding = self.embed_text(query, model_key=model_key)
        return embedding[0]  # Return single embedding

    def compute_similarity(self,
                          embedding1: np.ndarray,
                          embedding2: np.ndarray,
                          method: str = 'cosine') -> float:
        """
        Compute similarity between two embeddings

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            method: Similarity method ('cosine', 'euclidean', 'dot')

        Returns:
            Similarity score
        """
        if method == 'cosine':
            # Normalize vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return np.dot(embedding1, embedding2) / (norm1 * norm2)

        elif method == 'dot':
            return np.dot(embedding1, embedding2)

        elif method == 'euclidean':
            distance = np.linalg.norm(embedding1 - embedding2)
            return 1.0 / (1.0 + distance)  # Convert distance to similarity

        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available and loaded models"""
        info = {
            'available_models': {},
            'loaded_models': list(self.models.keys()),
            'section_mappings': self.section_model_mapping,
            'device': str(self.device),
            'cache_directory': str(self.cache_dir)
        }

        for key, config in self.model_configs.items():
            info['available_models'][key] = {
                'name': config['name'],
                'description': config['description'],
                'type': config['type'],
                'loaded': key in self.models,
                'embedding_dim': config['embedding_dim']
            }

        return info

    def benchmark_models(self, sample_texts: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark different models on sample texts

        Args:
            sample_texts: List of sample texts for benchmarking

        Returns:
            Performance metrics for each model
        """
        import time

        results = {}

        for model_key in self.model_configs.keys():
            try:
                start_time = time.time()

                # Try to load and use model
                embeddings = self.embed_text(sample_texts, model_key=model_key)

                end_time = time.time()

                results[model_key] = {
                    'success': True,
                    'embedding_time': end_time - start_time,
                    'embeddings_per_second': len(sample_texts) / (end_time - start_time),
                    'embedding_shape': embeddings.shape,
                    'avg_embedding_norm': float(np.mean(np.linalg.norm(embeddings, axis=1)))
                }

                logger.info(f"✅ {model_key}: {len(sample_texts)} texts in {end_time - start_time:.2f}s")

            except Exception as e:
                results[model_key] = {
                    'success': False,
                    'error': str(e)
                }
                logger.warning(f"⚠️ {model_key} benchmark failed: {e}")

        return results

    def save_embeddings(self, embeddings: Dict[str, np.ndarray], filepath: str):
        """Save embeddings to file"""
        try:
            # Convert to serializable format
            serializable_embeddings = {
                key: embedding.tolist() for key, embedding in embeddings.items()
            }

            with open(filepath, 'w') as f:
                json.dump(serializable_embeddings, f, indent=2)

            logger.info(f"💾 Saved {len(embeddings)} embeddings to {filepath}")

        except Exception as e:
            logger.error(f"❌ Failed to save embeddings: {e}")

    def load_embeddings(self, filepath: str) -> Dict[str, np.ndarray]:
        """Load embeddings from file"""
        try:
            with open(filepath, 'r') as f:
                serializable_embeddings = json.load(f)

            # Convert back to numpy arrays
            embeddings = {
                key: np.array(embedding) for key, embedding in serializable_embeddings.items()
            }

            logger.info(f"📂 Loaded {len(embeddings)} embeddings from {filepath}")
            return embeddings

        except Exception as e:
            logger.error(f"❌ Failed to load embeddings: {e}")
            return {}

    def cleanup_models(self):
        """Free up memory by clearing loaded models"""
        for model_key in list(self.models.keys()):
            if model_key != 'general':  # Keep fallback model
                del self.models[model_key]
                if model_key in self.tokenizers:
                    del self.tokenizers[model_key]

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("🧹 Cleaned up embedding models (kept general fallback)")


# Utility functions for embedding operations

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit vectors"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return embeddings / norms


def aggregate_embeddings(embeddings: List[np.ndarray], method: str = 'mean') -> np.ndarray:
    """
    Aggregate multiple embeddings into a single embedding

    Args:
        embeddings: List of embedding vectors
        method: Aggregation method ('mean', 'max', 'weighted_mean')

    Returns:
        Aggregated embedding vector
    """
    if not embeddings:
        raise ValueError("Cannot aggregate empty embedding list")

    embeddings_array = np.stack(embeddings)

    if method == 'mean':
        return np.mean(embeddings_array, axis=0)
    elif method == 'max':
        return np.max(embeddings_array, axis=0)
    elif method == 'weighted_mean':
        # Weight by embedding norm (stronger embeddings get more weight)
        weights = np.linalg.norm(embeddings_array, axis=1)
        weights = weights / np.sum(weights)
        return np.average(embeddings_array, axis=0, weights=weights)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def find_most_similar_sections(query_embedding: np.ndarray,
                              section_embeddings: Dict[str, np.ndarray],
                              top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Find most similar sections to a query

    Args:
        query_embedding: Query embedding vector
        section_embeddings: Dict mapping section IDs to embeddings
        top_k: Number of top results to return

    Returns:
        List of (section_id, similarity_score) tuples
    """
    similarities = []

    for section_id, section_embedding in section_embeddings.items():
        similarity = np.dot(query_embedding, section_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(section_embedding)
        )
        similarities.append((section_id, float(similarity)))

    # Sort by similarity score (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]