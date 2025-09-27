#!/usr/bin/env python3
"""
Section-Aware RAG Analyzer
Advanced scientific literature analysis with section-specific retrieval and reasoning
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

# Vector database imports
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available - falling back to FAISS")

# NLP and embedding imports
from sentence_transformers import SentenceTransformer
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# Fallback to existing FAISS system
import faiss

# Document parsing imports
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)


class SectionType(Enum):
    """Scientific paper section types"""
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
    FUTURE_WORK = "future_work"
    LIMITATIONS = "limitations"
    REFERENCES = "references"
    ACKNOWLEDGMENTS = "acknowledgments"
    SUPPLEMENTARY = "supplementary"
    UNKNOWN = "unknown"


class QueryType(Enum):
    """RAG query types for section-aware routing"""
    METHODOLOGICAL = "methodological"
    EMPIRICAL = "empirical"
    CONCEPTUAL = "conceptual"
    COMPARATIVE = "comparative"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    SYNTHESIS = "synthesis"
    EXPLORATION = "exploration"


@dataclass
class SectionContent:
    """Structured representation of a paper section"""
    text: str
    section_type: SectionType
    title: Optional[str] = None
    order: int = 0
    confidence: float = 1.0
    entities: List[str] = field(default_factory=list)
    figures: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    technical_terms: List[str] = field(default_factory=list)


@dataclass
class EnhancedPaperRepresentation:
    """Enhanced paper representation with section-aware structure"""
    paper_id: str
    title: str
    authors: List[str]
    abstract: Optional[str] = None
    sections: Dict[SectionType, SectionContent] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_mode: str = "unknown"


class SectionClassifier:
    """Classify text sections using rule-based + ML approaches"""

    def __init__(self):
        self.section_patterns = {
            SectionType.ABSTRACT: [
                r'\babstract\b', r'\bsummary\b'
            ],
            SectionType.INTRODUCTION: [
                r'\bintroduction\b', r'\bbackground\b', r'\boverview\b'
            ],
            SectionType.METHODS: [
                r'\bmethods?\b', r'\bmethodology\b', r'\bmaterials?\s+and\s+methods?\b',
                r'\bexperimental\s+procedures?\b', r'\bprotocols?\b'
            ],
            SectionType.RESULTS: [
                r'\bresults?\b', r'\bfindings?\b', r'\bobservations?\b'
            ],
            SectionType.DISCUSSION: [
                r'\bdiscussion\b', r'\binterpretation\b', r'\bimplications?\b'
            ],
            SectionType.CONCLUSION: [
                r'\bconclusions?\b', r'\bsummary\b', r'\bfinal\s+remarks?\b'
            ]
        }

        # Load spaCy model if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found - using rule-based classification only")

    def classify_section(self, text: str, title: str = "") -> Tuple[SectionType, float]:
        """
        Classify a text section with confidence score

        Args:
            text: Section text content
            title: Section title/header

        Returns:
            Tuple of (section_type, confidence_score)
        """
        import re

        # Combine title and text for classification
        full_text = f"{title} {text}".lower()

        best_match = SectionType.UNKNOWN
        best_confidence = 0.0

        # Rule-based classification
        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, full_text, re.IGNORECASE))
                if matches > 0:
                    # Calculate confidence based on match frequency and text length
                    confidence = min(matches / max(len(full_text.split()) / 100, 1), 1.0)
                    if confidence > best_confidence:
                        best_match = section_type
                        best_confidence = confidence

        # Enhanced classification with NLP features
        if self.nlp and best_confidence < 0.7:
            nlp_type, nlp_confidence = self._classify_with_nlp(text, title)
            if nlp_confidence > best_confidence:
                best_match = nlp_type
                best_confidence = nlp_confidence

        return best_match, best_confidence

    def _classify_with_nlp(self, text: str, title: str) -> Tuple[SectionType, float]:
        """NLP-based section classification"""
        # Simplified NLP classification based on linguistic features
        doc = self.nlp(text[:1000])  # Process first 1000 characters

        # Feature extraction
        method_verbs = ["perform", "conduct", "measure", "analyze", "collect", "assess"]
        result_verbs = ["show", "demonstrate", "reveal", "indicate", "suggest", "find"]
        discussion_verbs = ["interpret", "explain", "discuss", "consider", "argue"]

        verb_counts = {
            "method": sum(1 for token in doc if token.lemma_.lower() in method_verbs),
            "result": sum(1 for token in doc if token.lemma_.lower() in result_verbs),
            "discussion": sum(1 for token in doc if token.lemma_.lower() in discussion_verbs)
        }

        # Simple classification based on verb patterns
        max_type = max(verb_counts, key=verb_counts.get)
        max_count = verb_counts[max_type]

        if max_count > 0:
            confidence = min(max_count / max(len(list(doc)) / 50, 1), 0.8)
            if max_type == "method":
                return SectionType.METHODS, confidence
            elif max_type == "result":
                return SectionType.RESULTS, confidence
            elif max_type == "discussion":
                return SectionType.DISCUSSION, confidence

        return SectionType.UNKNOWN, 0.0


class SectionAwareRAGAnalyzer:
    """
    Advanced RAG system with section-aware retrieval and reasoning
    Extends existing FAISS-based system with ChromaDB and section intelligence
    """

    def __init__(self,
                 storage_path: str = "./section_aware_rag",
                 use_chromadb: bool = True,
                 openai_key: Optional[str] = None,
                 deepseek_key: Optional[str] = None,
                 embedding_model: str = "all-MiniLM-L6-v2"):

        self.storage_path = storage_path
        self.use_chromadb = use_chromadb and CHROMADB_AVAILABLE
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        self.deepseek_key = deepseek_key or os.getenv("DEEPSEEK_API_KEY")

        # Initialize components
        self.embedding_model = SentenceTransformer(embedding_model)
        self.section_classifier = SectionClassifier()

        # Storage backends
        self.chroma_client = None
        self.collections = {}
        self.faiss_indices = {}  # Fallback FAISS indices

        # BM25 for hybrid search
        self.bm25_indices = {}

        # Initialize storage
        self._initialize_storage()

        logger.info(f"🧠 Section-Aware RAG Analyzer initialized")
        logger.info(f"   Storage: {'ChromaDB' if self.use_chromadb else 'FAISS'}")
        logger.info(f"   Embedding Model: {embedding_model}")
        logger.info(f"   Section Classification: {'NLP + Rules' if SPACY_AVAILABLE else 'Rules Only'}")

    def _initialize_storage(self):
        """Initialize vector storage backend"""
        os.makedirs(self.storage_path, exist_ok=True)

        if self.use_chromadb:
            self._initialize_chromadb()
        else:
            self._initialize_faiss_fallback()

    def _initialize_chromadb(self):
        """Initialize ChromaDB collections"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=self.storage_path,
                settings=Settings(anonymized_telemetry=False)
            )

            # Create collections for different content types
            collection_configs = {
                "abstracts": "Paper abstracts with metadata",
                "sections": "Individual paper sections with section-type metadata",
                "chunks": "Semantic chunks with context metadata"
            }

            for name, description in collection_configs.items():
                try:
                    collection = self.chroma_client.get_collection(name)
                    logger.info(f"   📚 Loaded existing collection: {name}")
                except:
                    collection = self.chroma_client.create_collection(
                        name=name,
                        metadata={"description": description}
                    )
                    logger.info(f"   📚 Created new collection: {name}")

                self.collections[name] = collection

        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            logger.info("Falling back to FAISS storage")
            self.use_chromadb = False
            self._initialize_faiss_fallback()

    def _initialize_faiss_fallback(self):
        """Initialize FAISS indices as fallback"""
        logger.info("Initializing FAISS fallback storage")
        # This will use the existing FAISS logic from the original rag_analyzer.py
        pass

    def process_papers_with_sections(self, papers_data: List[Dict]) -> Dict[str, Any]:
        """
        Process papers with enhanced section-aware analysis

        Args:
            papers_data: List of paper dictionaries from existing pipeline

        Returns:
            Processing results with section analysis
        """
        logger.info(f"🔬 Processing {len(papers_data)} papers with section-aware analysis")

        processed_papers = []
        section_stats = {section.value: 0 for section in SectionType}

        for paper_data in papers_data:
            enhanced_paper = self._extract_and_classify_sections(paper_data)
            processed_papers.append(enhanced_paper)

            # Update section statistics
            for section_type in enhanced_paper.sections:
                section_stats[section_type.value] += 1

        # Store in vector database
        storage_results = self._store_enhanced_papers(processed_papers)

        # Build search indices
        search_results = self._build_search_indices(processed_papers)

        results = {
            "processed_papers": len(processed_papers),
            "section_statistics": section_stats,
            "storage_results": storage_results,
            "search_results": search_results,
            "total_sections": sum(section_stats.values()),
            "avg_sections_per_paper": sum(section_stats.values()) / len(processed_papers) if processed_papers else 0
        }

        logger.info(f"✅ Section-aware processing complete: {results['total_sections']} sections identified")
        return results

    def _extract_and_classify_sections(self, paper_data: Dict) -> EnhancedPaperRepresentation:
        """Extract and classify sections from paper data"""
        paper_id = paper_data.get('pmid', paper_data.get('id', 'unknown'))

        enhanced_paper = EnhancedPaperRepresentation(
            paper_id=paper_id,
            title=paper_data.get('title', ''),
            authors=paper_data.get('authors', []),
            abstract=paper_data.get('abstract'),
            metadata=paper_data,
            processing_mode=paper_data.get('processing_mode', 'full')
        )

        # Process abstract
        if enhanced_paper.abstract:
            abstract_section = SectionContent(
                text=enhanced_paper.abstract,
                section_type=SectionType.ABSTRACT,
                title="Abstract",
                confidence=1.0
            )
            enhanced_paper.sections[SectionType.ABSTRACT] = abstract_section

        # Process full-text sections if available
        if 'sections' in paper_data and paper_data['sections']:
            for section_title, section_text in paper_data['sections'].items():
                if section_text and len(section_text.strip()) > 50:
                    section_type, confidence = self.section_classifier.classify_section(
                        section_text, section_title
                    )

                    section_content = SectionContent(
                        text=section_text,
                        section_type=section_type,
                        title=section_title,
                        confidence=confidence
                    )

                    enhanced_paper.sections[section_type] = section_content

        # Extract full-text as single section if no sections available
        elif 'full_text' in paper_data and paper_data['full_text']:
            full_text = paper_data['full_text']
            if full_text and len(full_text.strip()) > 100:
                # Try to split into sections using simple heuristics
                sections = self._split_fulltext_into_sections(full_text)
                for section_type, section_text in sections.items():
                    section_content = SectionContent(
                        text=section_text,
                        section_type=section_type,
                        confidence=0.7  # Lower confidence for auto-split sections
                    )
                    enhanced_paper.sections[section_type] = section_content

        return enhanced_paper

    def _split_fulltext_into_sections(self, full_text: str) -> Dict[SectionType, str]:
        """Simple heuristic to split full-text into sections"""
        import re

        sections = {}

        # Common section headers
        section_headers = {
            SectionType.INTRODUCTION: [r'\n\s*(?:introduction|background)\s*\n', r'\n\s*1\.?\s*introduction'],
            SectionType.METHODS: [r'\n\s*(?:methods?|methodology|materials?\s+and\s+methods?)\s*\n'],
            SectionType.RESULTS: [r'\n\s*(?:results?|findings?)\s*\n'],
            SectionType.DISCUSSION: [r'\n\s*(?:discussion|interpretation)\s*\n'],
            SectionType.CONCLUSION: [r'\n\s*(?:conclusions?|summary)\s*\n']
        }

        # Find section boundaries
        boundaries = []
        for section_type, patterns in section_headers.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
                for match in matches:
                    boundaries.append((match.start(), section_type))

        # Sort boundaries by position
        boundaries.sort(key=lambda x: x[0])

        # Extract sections
        for i, (start_pos, section_type) in enumerate(boundaries):
            end_pos = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)
            section_text = full_text[start_pos:end_pos].strip()

            if len(section_text) > 100:  # Minimum section length
                sections[section_type] = section_text

        # If no sections found, treat as unknown section
        if not sections and len(full_text) > 100:
            sections[SectionType.UNKNOWN] = full_text

        return sections

    def _store_enhanced_papers(self, papers: List[EnhancedPaperRepresentation]) -> Dict[str, Any]:
        """Store enhanced papers in vector database"""
        if self.use_chromadb:
            return self._store_in_chromadb(papers)
        else:
            return self._store_in_faiss(papers)

    def _store_in_chromadb(self, papers: List[EnhancedPaperRepresentation]) -> Dict[str, Any]:
        """Store papers in ChromaDB collections"""
        results = {"abstracts": 0, "sections": 0, "chunks": 0}

        for paper in papers:
            # Store abstract
            if SectionType.ABSTRACT in paper.sections:
                abstract_section = paper.sections[SectionType.ABSTRACT]
                metadata = {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "content_type": "abstract",
                    "processing_mode": paper.processing_mode,
                    **paper.metadata
                }

                self.collections["abstracts"].add(
                    documents=[abstract_section.text],
                    metadatas=[metadata],
                    ids=[f"abstract_{paper.paper_id}"]
                )
                results["abstracts"] += 1

            # Store sections
            for section_type, section_content in paper.sections.items():
                if section_type != SectionType.ABSTRACT:
                    metadata = {
                        "paper_id": paper.paper_id,
                        "section_type": section_type.value,
                        "section_title": section_content.title or "",
                        "section_order": section_content.order,
                        "classification_confidence": section_content.confidence,
                        "content_type": "section",
                        "text_length": len(section_content.text),
                        **paper.metadata
                    }

                    self.collections["sections"].add(
                        documents=[section_content.text],
                        metadatas=[metadata],
                        ids=[f"section_{paper.paper_id}_{section_type.value}_{section_content.order}"]
                    )
                    results["sections"] += 1

                    # Create semantic chunks for long sections
                    if len(section_content.text) > 1000:
                        chunks = self._create_semantic_chunks(section_content.text, paper.paper_id, section_type)
                        for i, chunk in enumerate(chunks):
                            chunk_metadata = {
                                "paper_id": paper.paper_id,
                                "parent_section": section_type.value,
                                "chunk_index": i,
                                "content_type": "chunk",
                                **metadata
                            }

                            self.collections["chunks"].add(
                                documents=[chunk],
                                metadatas=[chunk_metadata],
                                ids=[f"chunk_{paper.paper_id}_{section_type.value}_{i}"]
                            )
                            results["chunks"] += 1

        return results

    def _create_semantic_chunks(self, text: str, paper_id: str, section_type: SectionType) -> List[str]:
        """Create semantically-aware chunks"""
        # Simple sentence-boundary chunking
        import re

        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) > 500:  # Chunk size limit
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if len(chunk) > 50]

    def _build_search_indices(self, papers: List[EnhancedPaperRepresentation]) -> Dict[str, Any]:
        """Build search indices for hybrid retrieval"""
        results = {}

        if BM25_AVAILABLE:
            # Build BM25 indices for keyword search
            abstracts_corpus = []
            sections_corpus = []

            for paper in papers:
                # Abstract corpus
                if SectionType.ABSTRACT in paper.sections:
                    abstracts_corpus.append(paper.sections[SectionType.ABSTRACT].text.split())

                # Sections corpus
                for section_type, section_content in paper.sections.items():
                    if section_type != SectionType.ABSTRACT:
                        sections_corpus.append(section_content.text.split())

            if abstracts_corpus:
                self.bm25_indices["abstracts"] = BM25Okapi(abstracts_corpus)
                results["bm25_abstracts"] = len(abstracts_corpus)

            if sections_corpus:
                self.bm25_indices["sections"] = BM25Okapi(sections_corpus)
                results["bm25_sections"] = len(sections_corpus)

        return results

    def section_aware_query(self,
                           query: str,
                           query_type: QueryType = QueryType.SYNTHESIS,
                           target_sections: List[SectionType] = None,
                           limit: int = 10,
                           hybrid_search: bool = True) -> Dict[str, Any]:
        """
        Perform section-aware RAG query

        Args:
            query: User query
            query_type: Type of query for routing optimization
            target_sections: Specific sections to search (None for all)
            limit: Maximum results to return
            hybrid_search: Use both vector and keyword search

        Returns:
            Enhanced query results with section-aware context
        """
        logger.info(f"🎯 Section-aware query: {query[:100]}...")

        # Query routing based on type
        search_strategy = self._determine_search_strategy(query_type, target_sections)

        # Retrieve relevant contexts
        contexts = self._retrieve_section_aware_contexts(
            query, search_strategy, limit, hybrid_search
        )

        # Generate response using retrieved contexts
        response = self._generate_section_aware_response(query, contexts, query_type)

        return {
            "query": query,
            "query_type": query_type.value,
            "search_strategy": search_strategy,
            "contexts": contexts,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }

    def _determine_search_strategy(self,
                                  query_type: QueryType,
                                  target_sections: List[SectionType] = None) -> Dict[str, Any]:
        """Determine optimal search strategy based on query type"""

        strategy = {
            "collections": ["abstracts", "sections"],
            "section_weights": {},
            "hybrid_ratio": 0.5  # Balance between vector and keyword search
        }

        # Query-type specific optimizations
        if query_type == QueryType.METHODOLOGICAL:
            strategy["section_weights"] = {
                SectionType.METHODS.value: 1.0,
                SectionType.METHODOLOGY.value: 1.0,
                SectionType.MATERIALS_METHODS.value: 1.0,
                SectionType.ABSTRACT.value: 0.3
            }
            strategy["hybrid_ratio"] = 0.7  # Favor keyword search for methods

        elif query_type == QueryType.EMPIRICAL:
            strategy["section_weights"] = {
                SectionType.RESULTS.value: 1.0,
                SectionType.FINDINGS.value: 1.0,
                SectionType.ABSTRACT.value: 0.5,
                SectionType.DISCUSSION.value: 0.3
            }

        elif query_type == QueryType.CONCEPTUAL:
            strategy["section_weights"] = {
                SectionType.INTRODUCTION.value: 1.0,
                SectionType.BACKGROUND.value: 1.0,
                SectionType.DISCUSSION.value: 0.8,
                SectionType.ABSTRACT.value: 0.6
            }
            strategy["hybrid_ratio"] = 0.3  # Favor semantic search for concepts

        elif query_type == QueryType.SYNTHESIS:
            # Balanced search across all sections
            strategy["section_weights"] = {section.value: 0.8 for section in SectionType}
            strategy["section_weights"][SectionType.ABSTRACT.value] = 1.0

        # Override with target sections if specified
        if target_sections:
            strategy["section_weights"] = {
                section.value: 1.0 for section in target_sections
            }

        return strategy

    def _retrieve_section_aware_contexts(self,
                                        query: str,
                                        strategy: Dict[str, Any],
                                        limit: int,
                                        hybrid_search: bool) -> List[Dict[str, Any]]:
        """Retrieve contexts using section-aware strategy"""

        all_contexts = []

        if self.use_chromadb:
            # ChromaDB vector search
            vector_contexts = self._chromadb_vector_search(query, strategy, limit)
            all_contexts.extend(vector_contexts)

            # BM25 keyword search (if available and hybrid enabled)
            if hybrid_search and BM25_AVAILABLE:
                keyword_contexts = self._bm25_keyword_search(query, strategy, limit // 2)
                all_contexts.extend(keyword_contexts)
        else:
            # Fallback FAISS search
            faiss_contexts = self._faiss_search(query, strategy, limit)
            all_contexts.extend(faiss_contexts)

        # Rank and deduplicate contexts
        ranked_contexts = self._rank_contexts_with_section_awareness(
            all_contexts, query, strategy, limit
        )

        return ranked_contexts

    def _chromadb_vector_search(self, query: str, strategy: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Perform vector search using ChromaDB"""
        contexts = []

        for collection_name in strategy["collections"]:
            if collection_name not in self.collections:
                continue

            collection = self.collections[collection_name]

            # Build metadata filter based on section weights
            where_conditions = []
            for section_type, weight in strategy["section_weights"].items():
                if weight > 0:
                    where_conditions.append({"section_type": {"$eq": section_type}})

            # Combine conditions with OR
            where_filter = {"$or": where_conditions} if where_conditions else None

            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=min(limit, 100),  # ChromaDB limit
                    where=where_filter
                )

                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    context = {
                        "text": doc,
                        "metadata": metadata,
                        "similarity_score": 1 - distance,  # Convert distance to similarity
                        "source": "chromadb_vector",
                        "collection": collection_name
                    }
                    contexts.append(context)

            except Exception as e:
                logger.error(f"ChromaDB search error: {e}")

        return contexts

    def _bm25_keyword_search(self, query: str, strategy: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Perform BM25 keyword search"""
        contexts = []
        query_tokens = query.lower().split()

        for index_name, bm25_index in self.bm25_indices.items():
            scores = bm25_index.get_scores(query_tokens)

            # Get top scoring documents
            top_indices = np.argsort(scores)[::-1][:limit]

            for idx in top_indices:
                if scores[idx] > 0:  # Only include documents with positive scores
                    context = {
                        "text": " ".join(bm25_index.corpus[idx]),
                        "metadata": {"index": index_name, "bm25_score": scores[idx]},
                        "similarity_score": min(scores[idx] / 10, 1.0),  # Normalize BM25 score
                        "source": "bm25_keyword",
                        "collection": index_name
                    }
                    contexts.append(context)

        return contexts

    def _rank_contexts_with_section_awareness(self,
                                            contexts: List[Dict[str, Any]],
                                            query: str,
                                            strategy: Dict[str, Any],
                                            limit: int) -> List[Dict[str, Any]]:
        """Rank contexts with section-aware scoring"""

        for context in contexts:
            # Base similarity score
            base_score = context.get("similarity_score", 0.0)

            # Section-aware boost
            section_type = context.get("metadata", {}).get("section_type", "unknown")
            section_weight = strategy["section_weights"].get(section_type, 0.5)

            # Content type boost
            content_type = context.get("metadata", {}).get("content_type", "unknown")
            content_boost = {
                "abstract": 1.0,
                "section": 0.9,
                "chunk": 0.8
            }.get(content_type, 0.7)

            # Combine scores
            final_score = base_score * section_weight * content_boost
            context["final_score"] = final_score

        # Sort by final score and deduplicate
        contexts.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        # Remove duplicates based on paper_id
        seen_papers = set()
        unique_contexts = []

        for context in contexts:
            paper_id = context.get("metadata", {}).get("paper_id")
            if paper_id not in seen_papers:
                unique_contexts.append(context)
                seen_papers.add(paper_id)
                if len(unique_contexts) >= limit:
                    break

        return unique_contexts

    def _generate_section_aware_response(self,
                                       query: str,
                                       contexts: List[Dict[str, Any]],
                                       query_type: QueryType) -> Dict[str, Any]:
        """Generate response with section-aware reasoning"""

        if not contexts:
            return {
                "answer": "No relevant information found in the analyzed literature.",
                "confidence": 0.0,
                "reasoning": "No matching documents in corpus",
                "sources": []
            }

        # Build section-organized context
        section_contexts = self._organize_contexts_by_section(contexts)

        # Generate LLM prompt with section awareness
        prompt = self._build_section_aware_prompt(query, section_contexts, query_type)

        # Generate response using available LLM
        llm_response = self._generate_llm_response(prompt, contexts)

        return llm_response

    def _organize_contexts_by_section(self, contexts: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Organize contexts by section type"""
        section_contexts = {}

        for context in contexts:
            section_type = context.get("metadata", {}).get("section_type", "unknown")
            if section_type not in section_contexts:
                section_contexts[section_type] = []
            section_contexts[section_type].append(context["text"])

        return section_contexts

    def _build_section_aware_prompt(self,
                                  query: str,
                                  section_contexts: Dict[str, List[str]],
                                  query_type: QueryType) -> str:
        """Build section-aware prompt for LLM"""

        prompt_parts = [
            f"You are analyzing scientific literature to answer the following {query_type.value} question:",
            f"QUESTION: {query}",
            "",
            "The information is organized by paper sections. Use this structure to provide a comprehensive answer:",
            ""
        ]

        # Add section-specific contexts
        for section_type, texts in section_contexts.items():
            if texts:
                prompt_parts.append(f"=== {section_type.upper()} SECTIONS ===")
                for i, text in enumerate(texts[:3]):  # Limit to top 3 per section
                    prompt_parts.append(f"{i+1}. {text[:500]}...")
                prompt_parts.append("")

        # Add query-type specific instructions
        instructions = {
            QueryType.METHODOLOGICAL: "Focus on methodological approaches, experimental designs, and techniques described in the Methods sections.",
            QueryType.EMPIRICAL: "Synthesize the key findings and empirical evidence from Results sections.",
            QueryType.CONCEPTUAL: "Provide conceptual analysis drawing from Introduction and Discussion sections.",
            QueryType.SYNTHESIS: "Synthesize information across all sections to provide a comprehensive overview."
        }

        prompt_parts.extend([
            "INSTRUCTIONS:",
            instructions.get(query_type, "Provide a comprehensive analysis based on the available literature."),
            "",
            "Please structure your response with:",
            "1. Direct answer to the question",
            "2. Supporting evidence from specific sections",
            "3. Any limitations or gaps in the available literature",
            "",
            "Answer:"
        ])

        return "\n".join(prompt_parts)

    def _generate_llm_response(self, prompt: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate LLM response (reuse existing logic from rag_analyzer.py)"""
        # This would integrate with the existing LLM generation logic
        # For now, return a placeholder
        return {
            "answer": "LLM integration would generate section-aware response here",
            "confidence": 0.8,
            "reasoning": "Section-aware analysis across multiple paper sections",
            "sources": [ctx.get("metadata", {}) for ctx in contexts[:5]]
        }

    def get_section_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored sections"""
        stats = {"total_papers": 0, "sections_by_type": {}, "collections": {}}

        if self.use_chromadb:
            for name, collection in self.collections.items():
                count = collection.count()
                stats["collections"][name] = count

                if name == "sections":
                    # Get section type breakdown
                    try:
                        # Note: This is a simplified count - ChromaDB doesn't have GROUP BY
                        stats["sections_by_type"] = {"estimated_total": count}
                    except:
                        pass

        return stats


# Convenience functions for integration with existing codebase
def create_section_aware_analyzer(storage_path: str = "./section_aware_rag", **kwargs) -> SectionAwareRAGAnalyzer:
    """Factory function to create section-aware analyzer"""
    return SectionAwareRAGAnalyzer(storage_path=storage_path, **kwargs)


def migrate_from_faiss(faiss_rag_analyzer, section_aware_analyzer: SectionAwareRAGAnalyzer):
    """Migrate data from existing FAISS-based RAG analyzer"""
    logger.info("Migration from FAISS to section-aware system would be implemented here")
    # Implementation would transfer existing vector indices and metadata
    pass