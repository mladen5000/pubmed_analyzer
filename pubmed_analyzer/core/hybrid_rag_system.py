#!/usr/bin/env python3
"""
Hybrid RAG System with DeepSeek Integration
Advanced retrieval system combining ChromaDB, section-aware search, and LLM reasoning
"""

import logging
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

# HTTP client for DeepSeek API
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import our existing components
from .section_aware_rag import SectionAwareRAGAnalyzer, QueryType, SectionType
from .chromadb_store import ScientificChromaStore
from .section_embeddings import MultiModelEmbedder
from ..models.paper import Paper

logger = logging.getLogger(__name__)


class ScoringStrategy(Enum):
    """Different scoring strategies for hybrid retrieval"""
    VECTOR_ONLY = "vector_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID_BALANCED = "hybrid_balanced"
    SECTION_WEIGHTED = "section_weighted"
    TEMPORAL_BOOST = "temporal_boost"
    CITATION_BOOST = "citation_boost"
    ADAPTIVE = "adaptive"


@dataclass
class QueryContext:
    """Enhanced query context for hybrid retrieval"""
    original_query: str
    processed_query: str
    query_type: QueryType
    target_sections: List[SectionType] = field(default_factory=list)
    temporal_preference: Optional[str] = None  # "recent", "historical", "all"
    domain_preference: Optional[str] = None
    methodology_focus: Optional[str] = None
    scoring_strategy: ScoringStrategy = ScoringStrategy.ADAPTIVE
    max_results: int = 10
    min_score_threshold: float = 0.3


@dataclass
class RetrievalResult:
    """Structured retrieval result with scoring breakdown"""
    content: str
    paper_id: str
    section_type: str
    title: str
    authors: List[str]
    journal: str
    year: int

    # Scoring breakdown
    vector_score: float
    keyword_score: float
    section_relevance_score: float
    temporal_score: float
    metadata_score: float
    final_score: float

    # Source attribution
    source_section: str
    confidence: float
    reasoning: str = ""


@dataclass
class LLMResponse:
    """Structured LLM response with attribution"""
    answer: str
    confidence: float
    sources: List[RetrievalResult]
    reasoning_steps: List[str]
    limitations: List[str]
    follow_up_questions: List[str]
    generation_time: float


class HybridScoringEngine:
    """Advanced scoring engine for multi-modal retrieval"""

    def __init__(self):
        self.scoring_weights = {
            ScoringStrategy.VECTOR_ONLY: {
                "vector": 1.0, "keyword": 0.0, "section": 0.0, "temporal": 0.0, "metadata": 0.0
            },
            ScoringStrategy.KEYWORD_ONLY: {
                "vector": 0.0, "keyword": 1.0, "section": 0.0, "temporal": 0.0, "metadata": 0.0
            },
            ScoringStrategy.HYBRID_BALANCED: {
                "vector": 0.4, "keyword": 0.3, "section": 0.2, "temporal": 0.05, "metadata": 0.05
            },
            ScoringStrategy.SECTION_WEIGHTED: {
                "vector": 0.3, "keyword": 0.2, "section": 0.4, "temporal": 0.05, "metadata": 0.05
            },
            ScoringStrategy.TEMPORAL_BOOST: {
                "vector": 0.3, "keyword": 0.2, "section": 0.2, "temporal": 0.25, "metadata": 0.05
            },
            ScoringStrategy.CITATION_BOOST: {
                "vector": 0.3, "keyword": 0.2, "section": 0.2, "temporal": 0.05, "metadata": 0.25
            },
            ScoringStrategy.ADAPTIVE: {
                "vector": 0.35, "keyword": 0.25, "section": 0.25, "temporal": 0.1, "metadata": 0.05
            }
        }

    def calculate_vector_score(self, similarity: float, query_embedding: np.ndarray,
                              result_embedding: np.ndarray) -> float:
        """Enhanced vector scoring with embedding quality assessment"""
        base_score = similarity

        # Boost for high-quality embeddings (higher norms often indicate more information)
        query_norm = np.linalg.norm(query_embedding)
        result_norm = np.linalg.norm(result_embedding)
        norm_boost = min(query_norm * result_norm / 100.0, 0.1)  # Small boost for strong embeddings

        return min(base_score + norm_boost, 1.0)

    def calculate_keyword_score(self, query: str, content: str, title: str = "") -> float:
        """BM25-inspired keyword scoring with title boost"""
        query_terms = set(query.lower().split())
        content_terms = content.lower().split()
        title_terms = title.lower().split()

        # Term frequency in content
        tf_scores = []
        for term in query_terms:
            tf_content = content_terms.count(term)
            tf_title = title_terms.count(term) * 2  # Title boost

            if tf_content + tf_title > 0:
                # Simple TF-IDF approximation
                tf_score = (tf_content + tf_title) / (len(content_terms) + len(title_terms) + 1)
                tf_scores.append(tf_score)

        return min(sum(tf_scores) / max(len(query_terms), 1), 1.0)

    def calculate_section_relevance_score(self, section_type: str, target_sections: List[SectionType],
                                        query_type: QueryType) -> float:
        """Section relevance based on query type and explicit preferences"""
        base_score = 0.5

        # Explicit section targeting
        if target_sections and any(s.value == section_type for s in target_sections):
            base_score = 1.0

        # Query type relevance mapping
        relevance_map = {
            QueryType.METHODOLOGICAL: {
                "methods": 1.0, "methodology": 1.0, "materials_methods": 1.0,
                "abstract": 0.3, "introduction": 0.2
            },
            QueryType.EMPIRICAL: {
                "results": 1.0, "findings": 1.0, "abstract": 0.5,
                "discussion": 0.3, "conclusion": 0.4
            },
            QueryType.CONCEPTUAL: {
                "introduction": 1.0, "background": 1.0, "discussion": 0.8,
                "abstract": 0.6, "conclusion": 0.5
            },
            QueryType.SYNTHESIS: {
                "abstract": 0.9, "discussion": 0.8, "conclusion": 0.8,
                "introduction": 0.6, "results": 0.7
            }
        }

        type_relevance = relevance_map.get(query_type, {}).get(section_type, base_score)
        return max(base_score, type_relevance)

    def calculate_temporal_score(self, year: int, temporal_preference: Optional[str]) -> float:
        """Temporal scoring based on publication year and preference"""
        current_year = datetime.now().year
        age = current_year - year

        if temporal_preference == "recent":
            # Exponential decay favoring recent papers
            return max(np.exp(-age / 3.0), 0.1)
        elif temporal_preference == "historical":
            # Slight preference for older, established papers
            return min(0.3 + age / 20.0, 1.0)
        else:  # "all" or None
            # Mild recency bias
            return max(1.0 - age / 20.0, 0.3)

    def calculate_metadata_score(self, metadata: Dict[str, Any], query_context: QueryContext) -> float:
        """Metadata-based scoring (journal impact, citation count, etc.)"""
        score = 0.5

        # Journal quality boost (simplified)
        high_impact_journals = {
            "nature", "science", "cell", "the lancet", "new england journal of medicine",
            "nature biotechnology", "nature medicine", "pnas"
        }
        journal = metadata.get("journal", "").lower()
        if any(hj in journal for hj in high_impact_journals):
            score += 0.2

        # Citation count boost (if available)
        citation_count = metadata.get("citation_count", 0)
        if citation_count > 0:
            score += min(np.log(citation_count + 1) / 10.0, 0.3)

        # Domain relevance
        if query_context.domain_preference:
            domain_keywords = query_context.domain_preference.lower().split()
            title_and_abstract = (metadata.get("title", "") + " " +
                                metadata.get("abstract", "")).lower()
            if any(keyword in title_and_abstract for keyword in domain_keywords):
                score += 0.1

        return min(score, 1.0)

    def calculate_hybrid_score(self, retrieval_data: Dict[str, Any],
                              query_context: QueryContext) -> float:
        """Calculate final hybrid score using weighted combination"""
        weights = self.scoring_weights[query_context.scoring_strategy]

        components = {
            "vector": retrieval_data.get("vector_score", 0.0),
            "keyword": retrieval_data.get("keyword_score", 0.0),
            "section": retrieval_data.get("section_relevance_score", 0.0),
            "temporal": retrieval_data.get("temporal_score", 0.0),
            "metadata": retrieval_data.get("metadata_score", 0.0)
        }

        # Weighted combination
        final_score = sum(weights[component] * score
                         for component, score in components.items())

        # Adaptive boosting based on query characteristics
        if query_context.scoring_strategy == ScoringStrategy.ADAPTIVE:
            final_score = self._apply_adaptive_boosting(final_score, components, query_context)

        return min(final_score, 1.0)

    def _apply_adaptive_boosting(self, base_score: float, components: Dict[str, float],
                               query_context: QueryContext) -> float:
        """Apply adaptive boosting based on query characteristics"""
        boost = 0.0

        # Boost for queries with strong vector similarity
        if components["vector"] > 0.8:
            boost += 0.05

        # Boost for exact keyword matches
        if components["keyword"] > 0.7:
            boost += 0.05

        # Boost for perfect section matches
        if components["section"] > 0.9:
            boost += 0.05

        # Boost for recent papers in methodology queries
        if (query_context.query_type == QueryType.METHODOLOGICAL and
            components["temporal"] > 0.7):
            boost += 0.03

        return base_score + boost


class DeepSeekClient:
    """DeepSeek API client with robust error handling and retry logic"""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1/chat/completions"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry logic"""
        session = requests.Session()

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

        return session

    async def generate_response(self, prompt: str, max_tokens: int = 1500,
                              temperature: float = 0.1) -> Dict[str, Any]:
        """Generate response from DeepSeek API"""
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }

        try:
            start_time = time.time()

            # Use asyncio to run in thread pool for non-blocking operation
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.session.post(self.base_url, json=payload, timeout=30)
            )

            generation_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "content": result["choices"][0]["message"]["content"],
                    "usage": result.get("usage", {}),
                    "generation_time": generation_time
                }
            else:
                logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}",
                    "generation_time": generation_time
                }

        except Exception as e:
            logger.error(f"DeepSeek client error: {e}")
            return {
                "success": False,
                "error": str(e),
                "generation_time": 0.0
            }


class HybridRAGSystem:
    """Main hybrid RAG system combining ChromaDB, section-aware search, and DeepSeek LLM"""

    def __init__(self,
                 chromadb_path: str = "./hybrid_rag_chromadb",
                 deepseek_api_key: Optional[str] = None,
                 embedding_cache_size: int = 1000):

        # Initialize components
        self.section_rag = SectionAwareRAGAnalyzer(storage_path=chromadb_path)
        self.scoring_engine = HybridScoringEngine()

        # DeepSeek client
        self.deepseek_key = deepseek_api_key or self._get_deepseek_key()
        self.deepseek_client = DeepSeekClient(self.deepseek_key) if self.deepseek_key else None

        # Performance optimization
        self.embedding_cache = {}
        self.embedding_cache_size = embedding_cache_size

        logger.info("🤖 Hybrid RAG System initialized")
        logger.info(f"   ChromaDB: {chromadb_path}")
        logger.info(f"   DeepSeek: {'✅ Connected' if self.deepseek_client else '❌ No API key'}")

    def _get_deepseek_key(self) -> Optional[str]:
        """Get DeepSeek API key from environment"""
        import os
        return os.getenv("DEEPSEEK_API_KEY")

    async def hybrid_retrieve(self, query_context: QueryContext) -> List[RetrievalResult]:
        """Advanced hybrid retrieval with multiple scoring strategies"""
        logger.info(f"🔍 Hybrid retrieval: {query_context.original_query}")

        # Get query embedding (with caching)
        query_embedding = self._get_cached_embedding(query_context.processed_query)

        # Retrieve from ChromaDB
        chromadb_results = await self._retrieve_from_chromadb(query_embedding, query_context)

        # Apply hybrid scoring
        scored_results = []
        for result in chromadb_results:
            scored_result = self._calculate_comprehensive_score(result, query_context, query_embedding)
            scored_results.append(scored_result)

        # Filter and sort
        filtered_results = [r for r in scored_results if r.final_score >= query_context.min_score_threshold]
        filtered_results.sort(key=lambda x: x.final_score, reverse=True)

        # Return top results
        return filtered_results[:query_context.max_results]

    def _get_cached_embedding(self, text: str) -> np.ndarray:
        """Get embedding with LRU cache"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        # Generate new embedding
        embedding = self.section_rag.embedding_model.encode([text])[0]

        # Cache management
        if len(self.embedding_cache) >= self.embedding_cache_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]

        self.embedding_cache[text] = embedding
        return embedding

    async def _retrieve_from_chromadb(self, query_embedding: np.ndarray,
                                     query_context: QueryContext) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from ChromaDB"""
        try:
            # Build section filters
            section_types = [s.value for s in query_context.target_sections] if query_context.target_sections else None

            # Build metadata filters
            filters = {}
            if query_context.temporal_preference == "recent":
                filters["year"] = {"$gte": datetime.now().year - 5}
            elif query_context.temporal_preference == "historical":
                filters["year"] = {"$lte": datetime.now().year - 10}

            # Query ChromaDB
            if hasattr(self.section_rag, 'chroma_client') and self.section_rag.chroma_client:
                results = self.section_rag._chromadb_vector_search(
                    query_context.processed_query,
                    {"collections": ["abstracts", "sections"], "section_weights": {}},
                    query_context.max_results * 2  # Get more for filtering
                )
            else:
                # Fallback to FAISS
                results = self.section_rag._faiss_search(
                    query_context.processed_query,
                    {"section_weights": {}},
                    query_context.max_results * 2
                )

            return results

        except Exception as e:
            logger.error(f"ChromaDB retrieval error: {e}")
            return []

    def _calculate_comprehensive_score(self, result: Dict[str, Any],
                                     query_context: QueryContext,
                                     query_embedding: np.ndarray) -> RetrievalResult:
        """Calculate comprehensive hybrid score for a retrieval result"""

        # Extract result data
        content = result.get("text", "")
        metadata = result.get("metadata", {})
        similarity = result.get("similarity_score", 0.0)

        # Calculate component scores
        vector_score = self.scoring_engine.calculate_vector_score(
            similarity, query_embedding, np.random.rand(384)  # Placeholder for result embedding
        )

        keyword_score = self.scoring_engine.calculate_keyword_score(
            query_context.processed_query, content, metadata.get("title", "")
        )

        section_relevance_score = self.scoring_engine.calculate_section_relevance_score(
            metadata.get("section_type", "unknown"),
            query_context.target_sections,
            query_context.query_type
        )

        temporal_score = self.scoring_engine.calculate_temporal_score(
            metadata.get("year", 2020), query_context.temporal_preference
        )

        metadata_score = self.scoring_engine.calculate_metadata_score(
            metadata, query_context
        )

        # Calculate final hybrid score
        scoring_data = {
            "vector_score": vector_score,
            "keyword_score": keyword_score,
            "section_relevance_score": section_relevance_score,
            "temporal_score": temporal_score,
            "metadata_score": metadata_score
        }

        final_score = self.scoring_engine.calculate_hybrid_score(scoring_data, query_context)

        # Create result object
        return RetrievalResult(
            content=content,
            paper_id=metadata.get("paper_id", "unknown"),
            section_type=metadata.get("section_type", "unknown"),
            title=metadata.get("title", ""),
            authors=metadata.get("authors", []),
            journal=metadata.get("journal", ""),
            year=metadata.get("year", 0),
            vector_score=vector_score,
            keyword_score=keyword_score,
            section_relevance_score=section_relevance_score,
            temporal_score=temporal_score,
            metadata_score=metadata_score,
            final_score=final_score,
            source_section=metadata.get("section_type", "unknown"),
            confidence=final_score,
            reasoning=f"Vector: {vector_score:.2f}, Keyword: {keyword_score:.2f}, Section: {section_relevance_score:.2f}"
        )

    async def answer_question(self, question: str,
                            query_type: QueryType = QueryType.SYNTHESIS,
                            scoring_strategy: ScoringStrategy = ScoringStrategy.ADAPTIVE,
                            max_context_length: int = 4000) -> LLMResponse:
        """Answer question using hybrid retrieval + DeepSeek generation"""

        if not self.deepseek_client:
            raise ValueError("DeepSeek API key not configured")

        logger.info(f"🤔 Answering question: {question}")

        # Create query context
        query_context = QueryContext(
            original_query=question,
            processed_query=self._preprocess_query(question),
            query_type=query_type,
            scoring_strategy=scoring_strategy,
            max_results=10
        )

        # Hybrid retrieval
        retrieval_results = await self.hybrid_retrieve(query_context)

        if not retrieval_results:
            return LLMResponse(
                answer="I couldn't find relevant information in the literature database to answer your question.",
                confidence=0.0,
                sources=[],
                reasoning_steps=["No relevant documents found"],
                limitations=["Limited database coverage"],
                follow_up_questions=[],
                generation_time=0.0
            )

        # Build context for LLM
        context = self._build_llm_context(retrieval_results, max_context_length)

        # Generate LLM prompt
        prompt = self._build_research_prompt(question, context, retrieval_results)

        # Generate response
        llm_result = await self.deepseek_client.generate_response(prompt)

        if llm_result["success"]:
            # Parse and structure response
            return self._parse_llm_response(
                llm_result["content"],
                retrieval_results,
                llm_result["generation_time"]
            )
        else:
            return LLMResponse(
                answer=f"Error generating response: {llm_result['error']}",
                confidence=0.0,
                sources=retrieval_results,
                reasoning_steps=[f"LLM error: {llm_result['error']}"],
                limitations=["LLM generation failed"],
                follow_up_questions=[],
                generation_time=llm_result["generation_time"]
            )

    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better retrieval"""
        # Simple preprocessing - can be enhanced
        processed = query.lower().strip()

        # Remove question words that don't help with retrieval
        question_words = ["what", "how", "why", "when", "where", "which", "who"]
        words = processed.split()
        filtered_words = [w for w in words if w not in question_words]

        return " ".join(filtered_words) if filtered_words else processed

    def _build_llm_context(self, results: List[RetrievalResult], max_length: int) -> str:
        """Build optimized context for LLM generation"""
        context_parts = []
        current_length = 0

        for i, result in enumerate(results):
            # Format source
            source_text = f"""
SOURCE {i+1} (Score: {result.final_score:.2f}):
Paper: {result.title}
Authors: {', '.join(result.authors[:3])}{'...' if len(result.authors) > 3 else ''}
Journal: {result.journal} ({result.year})
Section: {result.section_type}

Content: {result.content}

---"""

            if current_length + len(source_text) > max_length:
                break

            context_parts.append(source_text)
            current_length += len(source_text)

        return "\n".join(context_parts)

    def _build_research_prompt(self, question: str, context: str,
                             sources: List[RetrievalResult]) -> str:
        """Build research-focused prompt for LLM"""

        return f"""You are a scientific research assistant analyzing peer-reviewed literature. Answer the following research question based on the provided scientific sources.

RESEARCH QUESTION: {question}

SCIENTIFIC SOURCES:
{context}

INSTRUCTIONS:
1. Provide a comprehensive, evidence-based answer citing specific sources
2. Analyze methodology, findings, and limitations mentioned in the sources
3. Identify any conflicting results or viewpoints across papers
4. Mention confidence level and any limitations in the available evidence
5. Suggest follow-up research questions if appropriate

RESPONSE FORMAT:
## Answer
[Your comprehensive answer here]

## Key Evidence
[Bullet points of key supporting evidence with source citations]

## Methodology Analysis
[Analysis of research methods mentioned in the sources]

## Limitations & Gaps
[Any limitations in the evidence or research gaps identified]

## Confidence Assessment
[Your confidence in this answer: High/Medium/Low and why]

## Follow-up Questions
[Suggested research questions for further investigation]

Answer:"""

    def _parse_llm_response(self, response_text: str, sources: List[RetrievalResult],
                           generation_time: float) -> LLMResponse:
        """Parse and structure LLM response"""

        # Simple parsing - can be enhanced with more sophisticated NLP
        sections = response_text.split("##")

        answer = ""
        reasoning_steps = []
        limitations = []
        follow_up_questions = []

        for section in sections:
            section = section.strip()
            if section.startswith("Answer"):
                answer = section.replace("Answer", "").strip()
            elif section.startswith("Key Evidence"):
                reasoning_steps = [line.strip() for line in section.split("\n")[1:] if line.strip()]
            elif section.startswith("Limitations"):
                limitations = [line.strip() for line in section.split("\n")[1:] if line.strip()]
            elif section.startswith("Follow-up Questions"):
                follow_up_questions = [line.strip() for line in section.split("\n")[1:] if line.strip()]

        # Estimate confidence based on source quality and number
        avg_source_score = np.mean([s.final_score for s in sources]) if sources else 0.0
        source_count_factor = min(len(sources) / 5.0, 1.0)  # Normalize to 5 sources
        confidence = avg_source_score * source_count_factor

        return LLMResponse(
            answer=answer or response_text,  # Fallback to full response if parsing fails
            confidence=confidence,
            sources=sources,
            reasoning_steps=reasoning_steps,
            limitations=limitations,
            follow_up_questions=follow_up_questions,
            generation_time=generation_time
        )

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            "embedding_cache_size": len(self.embedding_cache),
            "deepseek_connected": self.deepseek_client is not None,
            "chromadb_stats": {},
            "section_rag_stats": {}
        }

        try:
            if hasattr(self.section_rag, 'get_section_statistics'):
                stats["section_rag_stats"] = self.section_rag.get_section_statistics()
        except Exception as e:
            logger.error(f"Error getting section RAG stats: {e}")

        return stats


# Convenience function for easy initialization
def create_hybrid_rag_system(chromadb_path: str = "./hybrid_rag_chromadb",
                            deepseek_api_key: Optional[str] = None) -> HybridRAGSystem:
    """Create and initialize hybrid RAG system"""
    return HybridRAGSystem(chromadb_path=chromadb_path, deepseek_api_key=deepseek_api_key)