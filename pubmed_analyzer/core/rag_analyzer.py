#!/usr/bin/env python3
"""
Enhanced RAG (Retrieval-Augmented Generation) Analyzer
Second-phase analysis with intelligent question-answering capabilities
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai
from datetime import datetime

logger = logging.getLogger(__name__)


class EnhancedRAGAnalyzer:
    """Enhanced RAG system for scientific literature analysis"""

    def __init__(self, openai_key: Optional[str] = None, deepseek_key: Optional[str] = None):
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        self.deepseek_key = deepseek_key or os.getenv("DEEPSEEK_API_KEY")

        # Initialize models
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Vector indices
        self.abstract_index = None
        self.fulltext_index = None
        self.section_index = None

        # Metadata for retrieval
        self.abstracts_metadata = []
        self.fulltext_metadata = []
        self.sections_metadata = []

        # Query templates
        self.query_templates = self._load_query_templates()

        logger.info("🤖 Enhanced RAG Analyzer initialized")
        if self.openai_key:
            logger.info("   OpenAI API: ✅")
        if self.deepseek_key:
            logger.info("   DeepSeek API: ✅")

    def build_vector_indices(self, papers_data: List[Dict]) -> Dict[str, Any]:
        """Build comprehensive vector indices for RAG"""
        logger.info("🔍 Building enhanced vector indices for RAG")

        results = {"indices_built": [], "metadata_counts": {}}

        # 1. Abstract index
        abstracts = []
        abstract_metadata = []

        for i, paper in enumerate(papers_data):
            abstract = paper.get('abstract', '')
            if abstract and len(abstract.strip()) > 50:
                abstracts.append(abstract)
                abstract_metadata.append({
                    "paper_id": i,
                    "title": paper.get('title', ''),
                    "authors": paper.get('authors', []),
                    "year": paper.get('year'),
                    "journal": paper.get('journal', ''),
                    "pmid": paper.get('pmid', ''),
                    "type": "abstract"
                })

        if abstracts:
            self._build_faiss_index(abstracts, abstract_metadata, "abstract")
            results["indices_built"].append("abstract")
            results["metadata_counts"]["abstracts"] = len(abstracts)

        # 2. Full-text index (chunked)
        fulltext_chunks = []
        fulltext_metadata = []

        for i, paper in enumerate(papers_data):
            fulltext = paper.get('full_text', '')
            if fulltext and len(fulltext.strip()) > 100:
                chunks = self._chunk_text(fulltext, chunk_size=500, overlap=50)
                for j, chunk in enumerate(chunks):
                    fulltext_chunks.append(chunk)
                    fulltext_metadata.append({
                        "paper_id": i,
                        "chunk_id": j,
                        "title": paper.get('title', ''),
                        "authors": paper.get('authors', []),
                        "year": paper.get('year'),
                        "journal": paper.get('journal', ''),
                        "pmid": paper.get('pmid', ''),
                        "type": "fulltext_chunk"
                    })

        if fulltext_chunks:
            self._build_faiss_index(fulltext_chunks, fulltext_metadata, "fulltext")
            results["indices_built"].append("fulltext")
            results["metadata_counts"]["fulltext_chunks"] = len(fulltext_chunks)

        # 3. Section-specific index
        sections = []
        section_metadata = []

        for i, paper in enumerate(papers_data):
            paper_sections = paper.get('sections', {})
            for section_name, section_text in paper_sections.items():
                if section_text and len(section_text.strip()) > 100:
                    sections.append(section_text)
                    section_metadata.append({
                        "paper_id": i,
                        "section": section_name,
                        "title": paper.get('title', ''),
                        "authors": paper.get('authors', []),
                        "year": paper.get('year'),
                        "journal": paper.get('journal', ''),
                        "pmid": paper.get('pmid', ''),
                        "type": "section"
                    })

        if sections:
            self._build_faiss_index(sections, section_metadata, "section")
            results["indices_built"].append("section")
            results["metadata_counts"]["sections"] = len(sections)

        logger.info(f"✅ Built {len(results['indices_built'])} vector indices")
        return results

    def interactive_analysis_session(self, papers_data: List[Dict]) -> Dict[str, Any]:
        """Run an interactive analysis session with predefined research questions"""
        logger.info("🎯 Starting enhanced RAG analysis session")

        # Build indices if not already built
        if not any([self.abstract_index, self.fulltext_index, self.section_index]):
            self.build_vector_indices(papers_data)

        # Predefined research questions
        research_questions = self._generate_research_questions(papers_data)

        session_results = {
            "session_timestamp": datetime.now().isoformat(),
            "questions_analyzed": len(research_questions),
            "results": []
        }

        for question_data in research_questions:
            logger.info(f"❓ Analyzing: {question_data['question'][:100]}...")

            try:
                answer = self.answer_research_question(
                    question=question_data["question"],
                    question_type=question_data["type"],
                    context_limit=5
                )

                session_results["results"].append({
                    "question": question_data["question"],
                    "question_type": question_data["type"],
                    "answer": answer,
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"❌ Failed to answer question: {e}")
                session_results["results"].append({
                    "question": question_data["question"],
                    "question_type": question_data["type"],
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

        return session_results

    def answer_research_question(self, question: str, question_type: str = "general",
                                context_limit: int = 5) -> Dict[str, Any]:
        """Answer a research question using RAG"""
        logger.info(f"🤔 Answering {question_type} question")

        # 1. Retrieve relevant context
        context_data = self._retrieve_relevant_context(question, question_type, context_limit)

        if not context_data["contexts"]:
            return {
                "answer": "I don't have sufficient relevant information to answer this question.",
                "confidence": 0.0,
                "sources": [],
                "reasoning": "No relevant documents found in the corpus."
            }

        # 2. Generate answer using LLM
        llm_response = self._generate_llm_answer(question, context_data, question_type)

        # 3. Post-process and validate answer
        final_answer = self._post_process_answer(llm_response, context_data)

        return final_answer

    def custom_query_analysis(self, custom_questions: List[str]) -> Dict[str, Any]:
        """Analyze custom user-provided questions"""
        logger.info(f"📝 Analyzing {len(custom_questions)} custom questions")

        results = {
            "custom_analysis_timestamp": datetime.now().isoformat(),
            "total_questions": len(custom_questions),
            "answers": []
        }

        for i, question in enumerate(custom_questions):
            logger.info(f"Processing question {i+1}/{len(custom_questions)}")

            try:
                # Classify question type automatically
                question_type = self._classify_question_type(question)

                answer = self.answer_research_question(
                    question=question,
                    question_type=question_type,
                    context_limit=7
                )

                results["answers"].append({
                    "question_id": i + 1,
                    "question": question,
                    "classified_type": question_type,
                    "answer": answer
                })

            except Exception as e:
                logger.error(f"❌ Error processing question {i+1}: {e}")
                results["answers"].append({
                    "question_id": i + 1,
                    "question": question,
                    "error": str(e)
                })

        return results

    def generate_research_insights(self, papers_data: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive research insights using RAG"""
        logger.info("💡 Generating research insights")

        insights = {
            "generation_timestamp": datetime.now().isoformat(),
            "corpus_size": len(papers_data),
            "insights": {}
        }

        # Define insight categories
        insight_questions = {
            "methodology_trends": "What are the main methodological approaches and trends in this research area?",
            "key_findings": "What are the most significant findings and discoveries reported in these papers?",
            "research_gaps": "What research gaps and limitations are frequently mentioned?",
            "future_directions": "What future research directions are suggested by these studies?",
            "technology_evolution": "How have the technologies and tools evolved in this field?",
            "collaboration_patterns": "What patterns of collaboration and authorship are evident?"
        }

        for category, question in insight_questions.items():
            try:
                answer = self.answer_research_question(
                    question=question,
                    question_type="analytical",
                    context_limit=10
                )
                insights["insights"][category] = answer

            except Exception as e:
                logger.error(f"❌ Failed to generate insight for {category}: {e}")
                insights["insights"][category] = {"error": str(e)}

        return insights

    def _build_faiss_index(self, texts: List[str], metadata: List[Dict], index_type: str):
        """Build FAISS index for given texts"""
        if not texts:
            return

        # Generate embeddings
        embeddings = self.sentence_model.encode(texts, show_progress_bar=False)

        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings.astype(np.float32))
        index.add(embeddings.astype(np.float32))

        # Store index and metadata
        if index_type == "abstract":
            self.abstract_index = index
            self.abstracts_metadata = metadata
        elif index_type == "fulltext":
            self.fulltext_index = index
            self.fulltext_metadata = metadata
        elif index_type == "section":
            self.section_index = index
            self.sections_metadata = metadata

        logger.info(f"✅ Built {index_type} index with {len(texts)} documents")

    def _retrieve_relevant_context(self, question: str, question_type: str, limit: int) -> Dict[str, Any]:
        """Retrieve relevant context for a question"""
        # Generate query embedding
        query_embedding = self.sentence_model.encode([question])
        faiss.normalize_L2(query_embedding.astype(np.float32))

        all_contexts = []

        # Search in different indices based on question type
        search_targets = self._determine_search_targets(question_type)

        for target in search_targets:
            if target == "abstract" and self.abstract_index:
                contexts = self._search_index(query_embedding, self.abstract_index,
                                            self.abstracts_metadata, limit//len(search_targets))
                all_contexts.extend(contexts)

            elif target == "fulltext" and self.fulltext_index:
                contexts = self._search_index(query_embedding, self.fulltext_index,
                                            self.fulltext_metadata, limit//len(search_targets))
                all_contexts.extend(contexts)

            elif target == "section" and self.section_index:
                contexts = self._search_index(query_embedding, self.section_index,
                                            self.sections_metadata, limit//len(search_targets))
                all_contexts.extend(contexts)

        # Rank and deduplicate
        ranked_contexts = self._rank_and_deduplicate_contexts(all_contexts, limit)

        return {
            "contexts": ranked_contexts,
            "total_found": len(all_contexts),
            "search_targets": search_targets
        }

    def _search_index(self, query_embedding: np.ndarray, index: faiss.Index,
                     metadata: List[Dict], limit: int) -> List[Dict]:
        """Search a specific FAISS index"""
        if index.ntotal == 0:
            return []

        k = min(limit, index.ntotal)
        scores, indices = index.search(query_embedding.astype(np.float32), k)

        contexts = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(metadata):
                context = metadata[idx].copy()
                context["similarity_score"] = float(score)
                contexts.append(context)

        return contexts

    def _generate_llm_answer(self, question: str, context_data: Dict, question_type: str) -> Dict[str, Any]:
        """Generate answer using available LLM"""
        # Prepare context text
        context_texts = []
        sources = []

        for ctx in context_data["contexts"]:
            if ctx.get("type") == "abstract":
                text = f"Abstract from '{ctx.get('title', 'Unknown')}': {ctx.get('abstract', '')}"
            elif ctx.get("type") == "fulltext_chunk":
                text = f"Excerpt from '{ctx.get('title', 'Unknown')}': {ctx.get('text', '')}"
            elif ctx.get("type") == "section":
                text = f"{ctx.get('section', 'Section')} from '{ctx.get('title', 'Unknown')}': {ctx.get('text', '')}"
            else:
                text = str(ctx.get("text", ""))

            context_texts.append(text)
            sources.append({
                "title": ctx.get("title", ""),
                "authors": ctx.get("authors", []),
                "year": ctx.get("year"),
                "journal": ctx.get("journal", ""),
                "pmid": ctx.get("pmid", ""),
                "similarity_score": ctx.get("similarity_score", 0.0)
            })

        combined_context = "\n\n".join(context_texts)

        # Prepare prompt
        prompt = self._build_rag_prompt(question, combined_context, question_type)

        # Try OpenAI first, then DeepSeek
        if self.openai_key:
            try:
                response = self._query_openai(prompt)
                return {
                    "answer": response,
                    "llm_used": "openai",
                    "sources": sources,
                    "context_length": len(combined_context)
                }
            except Exception as e:
                logger.warning(f"OpenAI query failed: {e}")

        if self.deepseek_key:
            try:
                response = self._query_deepseek(prompt)
                return {
                    "answer": response,
                    "llm_used": "deepseek",
                    "sources": sources,
                    "context_length": len(combined_context)
                }
            except Exception as e:
                logger.warning(f"DeepSeek query failed: {e}")

        # Fallback to simple extraction
        return {
            "answer": self._generate_extractive_answer(question, context_texts),
            "llm_used": "extractive_fallback",
            "sources": sources,
            "context_length": len(combined_context)
        }

    def _query_openai(self, prompt: str) -> str:
        """Query OpenAI API"""
        client = openai.OpenAI(api_key=self.openai_key)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert scientific literature analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )

        return response.choices[0].message.content

    def _query_deepseek(self, prompt: str) -> str:
        """Query DeepSeek API"""
        # Implement DeepSeek API call (similar structure to OpenAI)
        # This would need the actual DeepSeek API implementation
        return "DeepSeek integration would be implemented here"

    def _generate_research_questions(self, papers_data: List[Dict]) -> List[Dict]:
        """Generate intelligent research questions based on the corpus"""
        questions = []

        # Basic corpus analysis questions
        questions.extend([
            {
                "question": "What are the main research themes and topics covered in this literature collection?",
                "type": "thematic"
            },
            {
                "question": "What methodological approaches are most commonly used in these studies?",
                "type": "methodological"
            },
            {
                "question": "What are the key findings and conclusions reported across these papers?",
                "type": "findings"
            },
            {
                "question": "What limitations and challenges are frequently mentioned by researchers?",
                "type": "limitations"
            },
            {
                "question": "What future research directions are suggested in these papers?",
                "type": "future_work"
            }
        ])

        # Domain-specific questions based on keywords
        common_keywords = self._extract_common_keywords(papers_data)

        if common_keywords:
            questions.append({
                "question": f"How do the concepts of {', '.join(common_keywords[:3])} relate to each other in this research area?",
                "type": "conceptual"
            })

        # Temporal questions if years are available
        years = [p.get('year') for p in papers_data if p.get('year')]
        if years and len(set(years)) > 1:
            questions.append({
                "question": "How has this research area evolved over time?",
                "type": "temporal"
            })

        return questions

    def _load_query_templates(self) -> Dict[str, str]:
        """Load query templates for different question types"""
        return {
            "general": """Based on the following research papers, please answer this question: {question}

Context:
{context}

Please provide a comprehensive answer based on the evidence from these papers. Include specific details and cite relevant findings.""",

            "methodological": """Analyze the methodological approaches mentioned in these research papers to answer: {question}

Context:
{context}

Focus on research methods, experimental designs, data collection techniques, and analytical approaches.""",

            "findings": """Examine the key findings and results from these research papers to answer: {question}

Context:
{context}

Highlight important discoveries, outcomes, and conclusions reported in the studies.""",

            "analytical": """Provide an analytical synthesis of these research papers to address: {question}

Context:
{context}

Synthesize information across multiple papers to identify patterns, trends, and insights."""
        }

    def _build_rag_prompt(self, question: str, context: str, question_type: str) -> str:
        """Build RAG prompt based on question type"""
        template = self.query_templates.get(question_type, self.query_templates["general"])
        return template.format(question=question, context=context)

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Chunk text with overlap for better context retention"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + chunk_size//2, start + 100), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break

            chunks.append(text[start:end])
            start = end - overlap

            if end >= len(text):
                break

        return [chunk for chunk in chunks if len(chunk.strip()) > 50]

    def _classify_question_type(self, question: str) -> str:
        """Classify question type automatically"""
        question_lower = question.lower()

        if any(word in question_lower for word in ["method", "approach", "technique", "how"]):
            return "methodological"
        elif any(word in question_lower for word in ["finding", "result", "discover", "show"]):
            return "findings"
        elif any(word in question_lower for word in ["trend", "evolution", "change", "over time"]):
            return "temporal"
        elif any(word in question_lower for word in ["limitation", "challenge", "problem", "gap"]):
            return "limitations"
        elif any(word in question_lower for word in ["future", "next", "recommend", "suggest"]):
            return "future_work"
        else:
            return "general"

    def _determine_search_targets(self, question_type: str) -> List[str]:
        """Determine which indices to search based on question type"""
        if question_type == "methodological":
            return ["section", "fulltext", "abstract"]
        elif question_type == "findings":
            return ["abstract", "section", "fulltext"]
        elif question_type == "thematic":
            return ["abstract", "fulltext"]
        else:
            return ["abstract", "fulltext", "section"]

    def _rank_and_deduplicate_contexts(self, contexts: List[Dict], limit: int) -> List[Dict]:
        """Rank contexts by relevance and remove duplicates"""
        # Remove duplicates based on paper_id
        seen_papers = set()
        unique_contexts = []

        for ctx in contexts:
            paper_id = ctx.get("paper_id")
            if paper_id not in seen_papers:
                unique_contexts.append(ctx)
                seen_papers.add(paper_id)

        # Sort by similarity score
        unique_contexts.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)

        return unique_contexts[:limit]

    def _post_process_answer(self, llm_response: Dict, context_data: Dict) -> Dict[str, Any]:
        """Post-process and validate LLM answer"""
        answer = llm_response.get("answer", "")

        # Calculate confidence based on similarity scores and context quality
        similarity_scores = [ctx.get("similarity_score", 0) for ctx in context_data["contexts"]]
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0

        confidence = min(avg_similarity, 0.95)  # Cap at 95%

        return {
            "answer": answer,
            "confidence": float(confidence),
            "sources": llm_response.get("sources", []),
            "llm_used": llm_response.get("llm_used", "unknown"),
            "context_quality": {
                "avg_similarity": float(avg_similarity),
                "num_sources": len(context_data["contexts"]),
                "context_length": llm_response.get("context_length", 0)
            },
            "reasoning": f"Answer based on {len(context_data['contexts'])} relevant sources with average similarity of {avg_similarity:.2f}"
        }

    def _generate_extractive_answer(self, question: str, context_texts: List[str]) -> str:
        """Generate extractive answer when LLM is not available"""
        if not context_texts:
            return "No relevant information found to answer the question."

        # Simple extractive approach - find most relevant sentences
        question_words = set(question.lower().split())
        best_sentences = []

        for text in context_texts[:3]:  # Use top 3 contexts
            sentences = text.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:
                    sentence_words = set(sentence.lower().split())
                    overlap = len(question_words & sentence_words)
                    if overlap > 0:
                        best_sentences.append((sentence, overlap))

        # Sort by relevance and take top sentences
        best_sentences.sort(key=lambda x: x[1], reverse=True)

        if best_sentences:
            return '. '.join([sent[0] for sent in best_sentences[:3]]) + '.'
        else:
            return "Based on the available literature, " + context_texts[0][:200] + "..."

    def _extract_common_keywords(self, papers_data: List[Dict]) -> List[str]:
        """Extract common keywords from papers"""
        all_keywords = []

        for paper in papers_data:
            keywords = paper.get('keywords', [])
            if isinstance(keywords, list):
                all_keywords.extend(keywords)
            elif isinstance(keywords, str):
                all_keywords.extend(keywords.split(','))

        if all_keywords:
            from collections import Counter
            return [kw.strip() for kw, _ in Counter(all_keywords).most_common(10)]

        return []