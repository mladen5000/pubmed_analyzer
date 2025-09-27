#!/usr/bin/env python3
"""
Hierarchical Section-Aware Chunking for Scientific Papers
Optimized chunking strategies for different section types
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk"""
    chunk_id: str
    section_type: str
    chunk_order: int
    start_position: int
    end_position: int
    overlap_with_previous: int
    semantic_boundary: bool
    has_citations: bool
    has_figures: bool
    sentence_count: int
    paragraph_count: int
    confidence_score: float


@dataclass
class SectionChunk:
    """A chunk of text with section-aware metadata"""
    content: str
    metadata: ChunkMetadata
    section_context: str  # Surrounding context for better understanding
    citations: List[str]
    figures_tables: List[str]


class SectionAwareChunker:
    """
    Intelligent chunking system that adapts to different section types
    and preserves semantic boundaries within scientific papers
    """

    def __init__(self):
        """Initialize with section-specific chunking strategies"""

        # Chunking strategies optimized for different section types
        self.strategies = {
            'abstract': {
                'max_size': 512,
                'overlap': 0,  # Keep abstracts whole when possible
                'preserve_sentences': True,
                'preserve_paragraphs': True,
                'min_chunk_size': 100
            },
            'introduction': {
                'max_size': 800,
                'overlap': 100,
                'preserve_sentences': True,
                'preserve_paragraphs': True,
                'min_chunk_size': 200
            },
            'methods': {
                'max_size': 1000,
                'overlap': 150,
                'preserve_sentences': True,
                'preserve_paragraphs': False,  # Methods can be split mid-paragraph
                'min_chunk_size': 300
            },
            'results': {
                'max_size': 600,
                'overlap': 100,
                'preserve_sentences': True,
                'preserve_paragraphs': True,
                'min_chunk_size': 200
            },
            'discussion': {
                'max_size': 800,
                'overlap': 120,
                'preserve_sentences': True,
                'preserve_paragraphs': True,
                'min_chunk_size': 250
            },
            'conclusion': {
                'max_size': 400,
                'overlap': 50,
                'preserve_sentences': True,
                'preserve_paragraphs': True,
                'min_chunk_size': 100
            },
            'references': {
                'max_size': 200,
                'overlap': 0,  # References are independent
                'preserve_sentences': False,
                'preserve_paragraphs': False,
                'min_chunk_size': 50
            }
        }

        # Patterns for detecting semantic boundaries
        self.semantic_boundary_patterns = {
            'paragraph_break': r'\n\s*\n',
            'subsection_header': r'^[A-Z][^.]*:?\s*$',
            'numbered_point': r'^\d+[\.\)]\s+',
            'bullet_point': r'^[\-\*\•]\s+',
            'transition_words': r'\b(however|furthermore|moreover|additionally|in contrast|therefore|thus|consequently|nevertheless)\b'
        }

        # Citation and figure reference patterns
        self.citation_patterns = [
            r'\[(\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*)\]',
            r'\(([A-Za-z]+(?:\s+et\s+al\.?)?,?\s+\d{4}[a-z]?(?:;\s*[A-Za-z]+(?:\s+et\s+al\.?)?,?\s+\d{4}[a-z]?)*)\)',
        ]

        self.figure_table_patterns = [
            r'(?i)(?:figure|fig\.?)\s+(\d+[a-z]?)',
            r'(?i)(?:table|tab\.?)\s+(\d+[a-z]?)',
        ]

    def chunk_section(self,
                     section_content: str,
                     section_type: str,
                     section_metadata: Optional[Dict] = None) -> List[SectionChunk]:
        """
        Chunk a section using type-specific strategy

        Args:
            section_content: Text content of the section
            section_type: Type of section (abstract, introduction, etc.)
            section_metadata: Additional metadata about the section

        Returns:
            List of intelligently chunked sections
        """
        if not section_content.strip():
            return []

        # Get chunking strategy for this section type
        strategy = self.strategies.get(section_type, self.strategies['results'])

        logger.debug(f"🔄 Chunking {section_type} section ({len(section_content)} chars)")

        # If content is smaller than max size, return as single chunk
        if len(section_content) <= strategy['max_size']:
            chunk_metadata = ChunkMetadata(
                chunk_id=f"{section_type}_chunk_0",
                section_type=section_type,
                chunk_order=0,
                start_position=0,
                end_position=len(section_content),
                overlap_with_previous=0,
                semantic_boundary=True,
                has_citations=self._has_citations(section_content),
                has_figures=self._has_figures_tables(section_content),
                sentence_count=self._count_sentences(section_content),
                paragraph_count=self._count_paragraphs(section_content),
                confidence_score=1.0
            )

            return [SectionChunk(
                content=section_content,
                metadata=chunk_metadata,
                section_context=section_content[:200] + "...",
                citations=self._extract_citations(section_content),
                figures_tables=self._extract_figures_tables(section_content)
            )]

        # Perform intelligent chunking
        return self._intelligent_chunking(section_content, section_type, strategy, section_metadata)

    def _intelligent_chunking(self,
                            content: str,
                            section_type: str,
                            strategy: Dict,
                            section_metadata: Optional[Dict] = None) -> List[SectionChunk]:
        """Perform intelligent chunking with semantic boundary preservation"""

        chunks = []
        current_position = 0
        chunk_order = 0

        while current_position < len(content):
            # Calculate chunk boundaries
            chunk_start = max(0, current_position - strategy['overlap'])
            chunk_end = min(len(content), current_position + strategy['max_size'])

            # Find optimal chunk boundary
            optimal_end, is_semantic_boundary = self._find_optimal_boundary(
                content, chunk_start, chunk_end, strategy
            )

            # Extract chunk content
            chunk_content = content[chunk_start:optimal_end]

            # Skip if chunk is too small (unless it's the last chunk)
            if len(chunk_content.strip()) < strategy['min_chunk_size'] and optimal_end < len(content):
                current_position = optimal_end
                continue

            # Calculate overlap
            overlap = chunk_start - (current_position - strategy['overlap']) if chunk_order > 0 else 0

            # Create chunk metadata
            chunk_metadata = ChunkMetadata(
                chunk_id=f"{section_type}_chunk_{chunk_order}",
                section_type=section_type,
                chunk_order=chunk_order,
                start_position=chunk_start,
                end_position=optimal_end,
                overlap_with_previous=overlap,
                semantic_boundary=is_semantic_boundary,
                has_citations=self._has_citations(chunk_content),
                has_figures=self._has_figures_tables(chunk_content),
                sentence_count=self._count_sentences(chunk_content),
                paragraph_count=self._count_paragraphs(chunk_content),
                confidence_score=self._calculate_chunk_confidence(chunk_content, strategy)
            )

            # Create section context (surrounding text for better understanding)
            context_start = max(0, chunk_start - 100)
            context_end = min(len(content), optimal_end + 100)
            section_context = content[context_start:context_end]

            # Create chunk
            chunk = SectionChunk(
                content=chunk_content,
                metadata=chunk_metadata,
                section_context=section_context,
                citations=self._extract_citations(chunk_content),
                figures_tables=self._extract_figures_tables(chunk_content)
            )

            chunks.append(chunk)

            # Move to next chunk
            current_position = optimal_end
            chunk_order += 1

            # Safety check to prevent infinite loops
            if chunk_order > 100:
                logger.warning(f"⚠️ Chunking stopped at {chunk_order} chunks for safety")
                break

        logger.debug(f"✅ Created {len(chunks)} chunks for {section_type} section")
        return chunks

    def _find_optimal_boundary(self,
                             content: str,
                             start: int,
                             max_end: int,
                             strategy: Dict) -> Tuple[int, bool]:
        """Find the optimal boundary for chunking, preferring semantic boundaries"""

        # If we're at the end of content, return as-is
        if max_end >= len(content):
            return len(content), True

        # Try to find semantic boundaries in order of preference
        boundary_candidates = []

        # 1. Paragraph breaks (highest priority)
        if strategy.get('preserve_paragraphs', True):
            for match in re.finditer(self.semantic_boundary_patterns['paragraph_break'], content[start:max_end]):
                boundary_candidates.append((start + match.end(), 'paragraph', 1.0))

        # 2. Sentence endings
        if strategy.get('preserve_sentences', True):
            sentence_endings = r'[.!?]\s+'
            for match in re.finditer(sentence_endings, content[start:max_end]):
                # Prefer boundaries closer to the end but not too close to max
                distance_from_ideal = abs((start + match.end()) - (max_end * 0.8))
                score = 1.0 - (distance_from_ideal / (max_end - start))
                boundary_candidates.append((start + match.end(), 'sentence', score))

        # 3. Subsection headers
        for match in re.finditer(self.semantic_boundary_patterns['subsection_header'], content[start:max_end], re.MULTILINE):
            boundary_candidates.append((start + match.start(), 'subsection', 0.9))

        # 4. Numbered or bullet points
        for pattern_name in ['numbered_point', 'bullet_point']:
            pattern = self.semantic_boundary_patterns[pattern_name]
            for match in re.finditer(pattern, content[start:max_end], re.MULTILINE):
                boundary_candidates.append((start + match.start(), pattern_name, 0.7))

        # Select best boundary
        if boundary_candidates:
            # Filter candidates that are not too close to the start
            min_chunk_size = strategy.get('min_chunk_size', 100)
            valid_candidates = [
                (pos, type_, score) for pos, type_, score in boundary_candidates
                if pos - start >= min_chunk_size
            ]

            if valid_candidates:
                # Sort by score and select the best
                best_boundary = max(valid_candidates, key=lambda x: x[2])
                return best_boundary[0], True

        # No good semantic boundary found, use max_end
        return max_end, False

    def _has_citations(self, text: str) -> bool:
        """Check if text contains citations"""
        for pattern in self.citation_patterns:
            if re.search(pattern, text):
                return True
        return False

    def _has_figures_tables(self, text: str) -> bool:
        """Check if text contains figure or table references"""
        for pattern in self.figure_table_patterns:
            if re.search(pattern, text):
                return True
        return False

    def _extract_citations(self, text: str) -> List[str]:
        """Extract citation references from text"""
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        return list(set(citations))  # Remove duplicates

    def _extract_figures_tables(self, text: str) -> List[str]:
        """Extract figure and table references from text"""
        refs = []
        for pattern in self.figure_table_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            refs.extend([f"Figure/Table {match}" for match in matches])
        return list(set(refs))

    def _count_sentences(self, text: str) -> int:
        """Count sentences in text"""
        sentences = re.split(r'[.!?]+', text.strip())
        return len([s for s in sentences if s.strip()])

    def _count_paragraphs(self, text: str) -> int:
        """Count paragraphs in text"""
        paragraphs = re.split(r'\n\s*\n', text.strip())
        return len([p for p in paragraphs if p.strip()])

    def _calculate_chunk_confidence(self, chunk_content: str, strategy: Dict) -> float:
        """Calculate confidence score for chunk quality"""
        base_score = 0.8

        # Boost for optimal size
        optimal_size = strategy['max_size'] * 0.7
        size_score = 1.0 - abs(len(chunk_content) - optimal_size) / optimal_size
        size_bonus = min(0.1, size_score * 0.1)

        # Boost for complete sentences
        if chunk_content.strip().endswith(('.', '!', '?')):
            base_score += 0.1

        # Penalty for very short chunks
        if len(chunk_content) < strategy.get('min_chunk_size', 100):
            base_score -= 0.2

        # Boost for semantic completeness (paragraphs)
        paragraph_count = self._count_paragraphs(chunk_content)
        if paragraph_count >= 1:
            base_score += 0.05

        return min(1.0, max(0.0, base_score + size_bonus))

    def chunk_multiple_sections(self,
                               sections_data: Dict[str, str],
                               preserve_section_boundaries: bool = True) -> Dict[str, List[SectionChunk]]:
        """
        Chunk multiple sections while preserving relationships

        Args:
            sections_data: Dict mapping section types to content
            preserve_section_boundaries: Whether to prevent chunks from spanning sections

        Returns:
            Dict mapping section types to their chunks
        """
        all_chunks = {}

        # Process sections in a logical order
        section_order = ['abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion', 'references']

        for section_type in section_order:
            if section_type in sections_data:
                chunks = self.chunk_section(sections_data[section_type], section_type)
                if chunks:
                    all_chunks[section_type] = chunks

        # Process any remaining sections not in the standard order
        for section_type, content in sections_data.items():
            if section_type not in all_chunks:
                chunks = self.chunk_section(content, section_type)
                if chunks:
                    all_chunks[section_type] = chunks

        return all_chunks

    def get_chunk_summary(self, chunks: List[SectionChunk]) -> Dict[str, Any]:
        """Generate summary statistics for chunks"""
        if not chunks:
            return {}

        total_content_length = sum(len(chunk.content) for chunk in chunks)
        total_citations = sum(len(chunk.citations) for chunk in chunks)
        total_figures = sum(len(chunk.figures_tables) for chunk in chunks)

        return {
            'total_chunks': len(chunks),
            'total_content_length': total_content_length,
            'average_chunk_size': total_content_length // len(chunks),
            'total_citations': total_citations,
            'total_figures_tables': total_figures,
            'average_confidence': sum(chunk.metadata.confidence_score for chunk in chunks) / len(chunks),
            'semantic_boundaries': sum(1 for chunk in chunks if chunk.metadata.semantic_boundary),
            'chunks_with_citations': sum(1 for chunk in chunks if chunk.metadata.has_citations),
            'chunks_with_figures': sum(1 for chunk in chunks if chunk.metadata.has_figures)
        }

    def optimize_chunking_strategy(self,
                                 section_type: str,
                                 sample_content: str,
                                 target_chunk_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize chunking strategy for a specific section type based on sample content

        Args:
            section_type: Type of section to optimize for
            sample_content: Sample content to analyze
            target_chunk_count: Desired number of chunks (optional)

        Returns:
            Optimized strategy parameters
        """
        current_strategy = self.strategies.get(section_type, self.strategies['results']).copy()

        if target_chunk_count and target_chunk_count > 0:
            # Calculate optimal chunk size for target count
            content_length = len(sample_content)
            target_size = content_length // target_chunk_count

            # Adjust strategy
            current_strategy['max_size'] = min(1200, max(200, target_size))
            current_strategy['overlap'] = min(200, current_strategy['max_size'] // 5)

        # Test current strategy
        test_chunks = self.chunk_section(sample_content, section_type)
        summary = self.get_chunk_summary(test_chunks)

        return {
            'optimized_strategy': current_strategy,
            'test_results': summary,
            'recommendation': self._generate_strategy_recommendation(summary, section_type)
        }

    def _generate_strategy_recommendation(self, summary: Dict[str, Any], section_type: str) -> str:
        """Generate human-readable recommendation for chunking strategy"""
        recommendations = []

        avg_size = summary.get('average_chunk_size', 0)
        confidence = summary.get('average_confidence', 0)

        if avg_size < 200:
            recommendations.append("Consider increasing max_size for larger, more coherent chunks")
        elif avg_size > 1000:
            recommendations.append("Consider decreasing max_size for more focused chunks")

        if confidence < 0.7:
            recommendations.append("Consider adjusting semantic boundary preferences")

        semantic_ratio = summary.get('semantic_boundaries', 0) / max(1, summary.get('total_chunks', 1))
        if semantic_ratio < 0.5:
            recommendations.append("Consider strengthening semantic boundary detection")

        if not recommendations:
            recommendations.append("Current chunking strategy appears optimal")

        return " | ".join(recommendations)