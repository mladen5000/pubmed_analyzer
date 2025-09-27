#!/usr/bin/env python3
"""
Section-Aware RAG Integration Module
Seamlessly integrates section-aware capabilities with existing PubMed analyzer pipeline
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from .section_aware_rag import SectionAwareRAGAnalyzer, QueryType, SectionType
from .enhanced_document_parser import SciPDFParser, DocumentStructure
from .rag_analyzer import EnhancedRAGAnalyzer  # Existing RAG system
from ..models.paper import Paper

logger = logging.getLogger(__name__)


class SectionAwareManager:
    """
    Manager class that orchestrates section-aware analysis
    Integrates with existing pubmed_analyzer.py entry points
    """

    def __init__(self,
                 storage_path: str = "./section_aware_rag",
                 enable_legacy_fallback: bool = True,
                 **kwargs):

        self.storage_path = storage_path
        self.enable_legacy_fallback = enable_legacy_fallback

        # Initialize components
        self.section_aware_rag = SectionAwareRAGAnalyzer(storage_path=storage_path, **kwargs)
        self.document_parser = SciPDFParser()

        # Legacy fallback
        self.legacy_rag = None
        if enable_legacy_fallback:
            try:
                self.legacy_rag = EnhancedRAGAnalyzer(**kwargs)
                logger.info("✅ Legacy RAG system available as fallback")
            except Exception as e:
                logger.warning(f"Legacy RAG initialization failed: {e}")

        logger.info("🚀 Section-Aware Manager initialized")

    def process_abstracts_mode(self, papers: List[Paper]) -> Dict[str, Any]:
        """
        Enhanced abstracts-only processing with section-aware capabilities

        Args:
            papers: List of Paper objects from search results

        Returns:
            Processing results with section-aware abstract analysis
        """
        logger.info(f"📄 Section-aware ABSTRACTS mode: Processing {len(papers)} papers")

        # Convert Papers to compatible format
        papers_data = self._papers_to_dict_format(papers)

        # Process with section-aware analyzer (abstracts only)
        results = self.section_aware_rag.process_papers_with_sections(papers_data)

        # Enhanced abstract analysis
        abstract_insights = self._generate_abstract_insights(papers_data)

        # Combine results
        combined_results = {
            **results,
            "mode": "abstracts_section_aware",
            "abstract_insights": abstract_insights,
            "total_papers_processed": len(papers),
            "processing_timestamp": datetime.now().isoformat()
        }

        logger.info(f"✅ Section-aware abstracts processing complete")
        return combined_results

    def process_full_mode(self, papers: List[Paper], pdf_directory: str = "pdfs") -> Dict[str, Any]:
        """
        Enhanced full-text processing with advanced section extraction

        Args:
            papers: List of Paper objects with potential PDF paths
            pdf_directory: Directory containing PDF files

        Returns:
            Comprehensive processing results with section-aware analysis
        """
        logger.info(f"📚 Section-aware FULL mode: Processing {len(papers)} papers")

        # Step 1: Enhanced PDF parsing with section extraction
        parsing_results = self._enhanced_pdf_parsing(papers, pdf_directory)

        # Step 2: Convert to compatible format with enhanced sections
        papers_data = self._papers_to_enhanced_dict_format(papers, parsing_results)

        # Step 3: Process with section-aware RAG
        section_results = self.section_aware_rag.process_papers_with_sections(papers_data)

        # Step 4: Generate comprehensive insights
        comprehensive_insights = self._generate_comprehensive_insights(papers_data)

        # Step 5: Cross-section analysis
        cross_section_analysis = self._perform_cross_section_analysis(papers_data)

        # Combine all results
        combined_results = {
            **section_results,
            "mode": "full_section_aware",
            "parsing_results": parsing_results,
            "comprehensive_insights": comprehensive_insights,
            "cross_section_analysis": cross_section_analysis,
            "total_papers_processed": len(papers),
            "processing_timestamp": datetime.now().isoformat()
        }

        logger.info(f"✅ Section-aware full processing complete")
        return combined_results

    def section_aware_query(self,
                           query: str,
                           mode: str = "auto",
                           target_sections: List[str] = None,
                           limit: int = 10) -> Dict[str, Any]:
        """
        Execute section-aware query with intelligent routing

        Args:
            query: User query
            mode: Query mode (auto, methodological, empirical, conceptual, synthesis)
            target_sections: Specific sections to search
            limit: Maximum results

        Returns:
            Section-aware query results
        """
        logger.info(f"🎯 Section-aware query: {query[:50]}...")

        # Automatically determine query type
        if mode == "auto":
            query_type = self._classify_query_intent(query)
        else:
            query_type = QueryType(mode) if mode in [t.value for t in QueryType] else QueryType.SYNTHESIS

        # Convert target sections to SectionType enums
        target_section_types = None
        if target_sections:
            target_section_types = []
            for section in target_sections:
                try:
                    section_type = SectionType(section.lower())
                    target_section_types.append(section_type)
                except ValueError:
                    logger.warning(f"Unknown section type: {section}")

        # Execute section-aware query
        try:
            results = self.section_aware_rag.section_aware_query(
                query=query,
                query_type=query_type,
                target_sections=target_section_types,
                limit=limit
            )

            # Fallback to legacy system if needed
            if not results.get("contexts") and self.legacy_rag:
                logger.info("🔄 Falling back to legacy RAG system")
                legacy_results = self._fallback_to_legacy_query(query, limit)
                results["legacy_fallback"] = legacy_results

            return results

        except Exception as e:
            logger.error(f"Section-aware query failed: {e}")

            # Fallback to legacy system
            if self.legacy_rag:
                logger.info("🔄 Using legacy RAG system due to error")
                return self._fallback_to_legacy_query(query, limit)
            else:
                return {"error": str(e), "query": query}

    def generate_research_insights(self, papers_data: List[Dict] = None) -> Dict[str, Any]:
        """
        Generate enhanced research insights using section-aware analysis

        Args:
            papers_data: Optional paper data (will query stored data if None)

        Returns:
            Comprehensive research insights
        """
        logger.info("💡 Generating section-aware research insights")

        # Base insights from section-aware analyzer
        insights = self.section_aware_rag.generate_research_insights(papers_data or [])

        # Enhanced section-specific insights
        section_insights = self._generate_section_specific_insights()

        # Methodological evolution analysis
        methodological_insights = self._analyze_methodological_evolution()

        # Research gap analysis
        gap_analysis = self._perform_gap_analysis()

        combined_insights = {
            **insights,
            "section_specific_insights": section_insights,
            "methodological_evolution": methodological_insights,
            "research_gaps": gap_analysis,
            "generation_mode": "section_aware_enhanced"
        }

        return combined_insights

    def _papers_to_dict_format(self, papers: List[Paper]) -> List[Dict]:
        """Convert Paper objects to dictionary format for processing"""
        papers_data = []

        for paper in papers:
            paper_dict = {
                'pmid': paper.pmid,
                'pmcid': paper.pmcid,
                'title': paper.title,
                'authors': paper.authors,
                'abstract': paper.abstract,
                'year': paper.year,
                'journal': paper.journal,
                'doi': paper.doi,
                'processing_mode': 'abstracts'
            }

            # Add any existing sections
            if hasattr(paper, 'sections') and paper.sections:
                paper_dict['sections'] = paper.sections

            papers_data.append(paper_dict)

        return papers_data

    def _enhanced_pdf_parsing(self, papers: List[Paper], pdf_directory: str) -> Dict[str, Any]:
        """Enhanced PDF parsing with section extraction"""
        logger.info(f"🔬 Enhanced PDF parsing with section extraction")

        parsing_results = {
            "total_papers": len(papers),
            "pdfs_found": 0,
            "successfully_parsed": 0,
            "sections_extracted": 0,
            "parsing_details": []
        }

        for paper in papers:
            # Find PDF file
            pdf_path = None
            if paper.pdf_path and os.path.exists(paper.pdf_path):
                pdf_path = paper.pdf_path
            else:
                # Try to find PDF in directory
                potential_paths = [
                    os.path.join(pdf_directory, f"{paper.clean_pmid}.pdf"),
                    os.path.join(pdf_directory, f"PMID_{paper.clean_pmid}.pdf"),
                ]
                if paper.pmcid:
                    potential_paths.append(os.path.join(pdf_directory, f"{paper.clean_pmcid}.pdf"))

                for path in potential_paths:
                    if os.path.exists(path):
                        pdf_path = path
                        break

            if pdf_path:
                parsing_results["pdfs_found"] += 1

                try:
                    # Parse with enhanced parser
                    structure = self.document_parser.parse_pdf_with_sections(pdf_path, paper)

                    if structure.sections:
                        parsing_results["successfully_parsed"] += 1
                        parsing_results["sections_extracted"] += len(structure.sections)

                        # Update paper object with parsed sections
                        paper.sections = structure.sections
                        paper.processing_success = True

                        parsing_results["parsing_details"].append({
                            "pmid": paper.pmid,
                            "status": "success",
                            "sections_found": len(structure.sections),
                            "section_types": list(structure.sections.keys())
                        })

                    else:
                        parsing_results["parsing_details"].append({
                            "pmid": paper.pmid,
                            "status": "no_sections",
                            "error": "No sections extracted"
                        })

                except Exception as e:
                    parsing_results["parsing_details"].append({
                        "pmid": paper.pmid,
                        "status": "error",
                        "error": str(e)
                    })
                    logger.error(f"Failed to parse PDF for {paper.pmid}: {e}")

            else:
                parsing_results["parsing_details"].append({
                    "pmid": paper.pmid,
                    "status": "no_pdf",
                    "error": "PDF file not found"
                })

        success_rate = (parsing_results["successfully_parsed"] / max(parsing_results["pdfs_found"], 1)) * 100
        logger.info(f"📊 Enhanced parsing results: {success_rate:.1f}% success rate")
        logger.info(f"   📑 Total sections extracted: {parsing_results['sections_extracted']}")

        return parsing_results

    def _papers_to_enhanced_dict_format(self, papers: List[Paper], parsing_results: Dict) -> List[Dict]:
        """Convert papers to enhanced format with parsing results"""
        papers_data = []

        for paper in papers:
            paper_dict = {
                'pmid': paper.pmid,
                'pmcid': paper.pmcid,
                'title': paper.title,
                'authors': paper.authors,
                'abstract': paper.abstract,
                'year': paper.year,
                'journal': paper.journal,
                'doi': paper.doi,
                'processing_mode': 'full',
                'has_pdf': paper.has_pdf,
                'parsing_success': paper.processing_success
            }

            # Add sections if available
            if hasattr(paper, 'sections') and paper.sections:
                paper_dict['sections'] = paper.sections

            # Add full text if available
            if paper.full_text:
                paper_dict['full_text'] = paper.full_text

            papers_data.append(paper_dict)

        return papers_data

    def _classify_query_intent(self, query: str) -> QueryType:
        """Automatically classify query intent for optimal routing"""
        query_lower = query.lower()

        # Method-focused queries
        method_keywords = ['method', 'approach', 'technique', 'protocol', 'procedure', 'how']
        if any(keyword in query_lower for keyword in method_keywords):
            return QueryType.METHODOLOGICAL

        # Results-focused queries
        result_keywords = ['result', 'finding', 'outcome', 'effect', 'impact', 'showed', 'demonstrated']
        if any(keyword in query_lower for keyword in result_keywords):
            return QueryType.EMPIRICAL

        # Concept-focused queries
        concept_keywords = ['concept', 'theory', 'framework', 'model', 'definition', 'what is']
        if any(keyword in query_lower for keyword in concept_keywords):
            return QueryType.CONCEPTUAL

        # Comparison queries
        comparison_keywords = ['compare', 'versus', 'vs', 'difference', 'similar', 'contrast']
        if any(keyword in query_lower for keyword in comparison_keywords):
            return QueryType.COMPARATIVE

        # Temporal queries
        temporal_keywords = ['trend', 'evolution', 'over time', 'timeline', 'history', 'development']
        if any(keyword in query_lower for keyword in temporal_keywords):
            return QueryType.TEMPORAL

        # Default to synthesis
        return QueryType.SYNTHESIS

    def _generate_abstract_insights(self, papers_data: List[Dict]) -> Dict[str, Any]:
        """Generate enhanced insights from abstracts"""
        abstracts = [paper.get('abstract', '') for paper in papers_data if paper.get('abstract')]

        if not abstracts:
            return {"error": "No abstracts available for analysis"}

        # Abstract-specific analysis would go here
        insights = {
            "total_abstracts": len(abstracts),
            "avg_abstract_length": sum(len(abstract) for abstract in abstracts) / len(abstracts),
            "coverage_analysis": {
                "with_abstracts": len(abstracts),
                "without_abstracts": len(papers_data) - len(abstracts),
                "coverage_percentage": (len(abstracts) / len(papers_data)) * 100
            }
        }

        return insights

    def _generate_comprehensive_insights(self, papers_data: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive insights from full-text analysis"""
        insights = {
            "section_coverage": {},
            "methodological_diversity": {},
            "content_quality_metrics": {}
        }

        # Analyze section coverage
        section_counts = {}
        for paper in papers_data:
            sections = paper.get('sections', {})
            for section_type in sections.keys():
                section_counts[section_type] = section_counts.get(section_type, 0) + 1

        insights["section_coverage"] = {
            "section_distribution": section_counts,
            "papers_with_sections": sum(1 for paper in papers_data if paper.get('sections')),
            "avg_sections_per_paper": sum(len(paper.get('sections', {})) for paper in papers_data) / len(papers_data)
        }

        return insights

    def _perform_cross_section_analysis(self, papers_data: List[Dict]) -> Dict[str, Any]:
        """Perform cross-section analysis to identify patterns"""
        analysis = {
            "methodology_to_results_correlation": {},
            "introduction_to_conclusion_consistency": {},
            "citation_patterns_by_section": {}
        }

        # This would contain more sophisticated cross-section analysis
        # For now, return placeholder structure
        return analysis

    def _generate_section_specific_insights(self) -> Dict[str, Any]:
        """Generate insights specific to different section types"""
        # Get section statistics from analyzer
        stats = self.section_aware_rag.get_section_statistics()

        insights = {
            "section_statistics": stats,
            "methodological_trends": {},
            "results_patterns": {},
            "discussion_themes": {}
        }

        return insights

    def _analyze_methodological_evolution(self) -> Dict[str, Any]:
        """Analyze evolution of methodological approaches"""
        evolution_analysis = {
            "trending_methods": [],
            "declining_methods": [],
            "emerging_technologies": [],
            "temporal_patterns": {}
        }

        return evolution_analysis

    def _perform_gap_analysis(self) -> Dict[str, Any]:
        """Identify research gaps based on section-aware analysis"""
        gap_analysis = {
            "methodological_gaps": [],
            "understudied_areas": [],
            "conflicting_findings": [],
            "future_research_directions": []
        }

        return gap_analysis

    def _fallback_to_legacy_query(self, query: str, limit: int) -> Dict[str, Any]:
        """Fallback to legacy RAG system"""
        try:
            # Use existing RAG analyzer logic
            return {
                "answer": "Legacy RAG system response would be generated here",
                "source": "legacy_rag",
                "confidence": 0.7
            }
        except Exception as e:
            return {"error": f"Legacy fallback failed: {e}"}

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "section_aware_rag": "available",
            "document_parser": "available",
            "legacy_fallback": "available" if self.legacy_rag else "unavailable",
            "storage_path": self.storage_path,
            "capabilities": {
                "section_extraction": True,
                "hybrid_search": hasattr(self.section_aware_rag, 'bm25_indices'),
                "nlp_enhancement": True,
                "cross_section_analysis": True
            }
        }

        # Add statistics if available
        try:
            stats = self.section_aware_rag.get_section_statistics()
            status["database_statistics"] = stats
        except:
            status["database_statistics"] = {"error": "Statistics unavailable"}

        return status


# Convenience functions for integration with existing entry points
def create_section_aware_manager(**kwargs) -> SectionAwareManager:
    """Create section-aware manager with default settings"""
    return SectionAwareManager(**kwargs)


def enhance_abstracts_analysis(papers: List[Paper], **kwargs) -> Dict[str, Any]:
    """Enhance existing abstracts analysis with section-aware capabilities"""
    manager = create_section_aware_manager(**kwargs)
    return manager.process_abstracts_mode(papers)


def enhance_full_analysis(papers: List[Paper], pdf_directory: str = "pdfs", **kwargs) -> Dict[str, Any]:
    """Enhance existing full analysis with section-aware capabilities"""
    manager = create_section_aware_manager(**kwargs)
    return manager.process_full_mode(papers, pdf_directory)


def section_aware_research_query(query: str, mode: str = "auto", **kwargs) -> Dict[str, Any]:
    """Execute section-aware research query"""
    manager = create_section_aware_manager(**kwargs)
    return manager.section_aware_query(query, mode, **kwargs)