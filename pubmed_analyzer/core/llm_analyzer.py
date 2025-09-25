#!/usr/bin/env python3
"""
Comprehensive LLM Analysis Module
Advanced batch analysis, structured insights, and response visualization
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import openai
import requests
from dataclasses import dataclass
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


@dataclass
class LLMAnalysisResult:
    """Structured LLM analysis result"""
    paper_id: str
    title: str
    analysis_type: str
    prompt_used: str
    response: str
    structured_scores: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: str
    processing_time: float
    llm_used: str


class ComprehensiveLLMAnalyzer:
    """Advanced LLM analyzer with batch processing and structured insights"""

    def __init__(self, openai_key: Optional[str] = None, deepseek_key: Optional[str] = None):
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        self.deepseek_key = deepseek_key or os.getenv("DEEPSEEK_API_KEY")

        # Analysis templates
        self.analysis_templates = self._load_analysis_templates()

        # Results storage
        self.analysis_results: List[LLMAnalysisResult] = []
        self.batch_results: Dict[str, Any] = {}

        logger.info("🧠 Comprehensive LLM Analyzer initialized")
        if self.openai_key:
            logger.info("   OpenAI API: ✅")
        if self.deepseek_key:
            logger.info("   DeepSeek API: ✅")

    async def comprehensive_batch_analysis(self, papers_data: List[Dict],
                                         analysis_types: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive batch analysis on all papers"""
        if analysis_types is None:
            analysis_types = [
                "scientific_quality", "methodology_assessment", "innovation_score",
                "research_impact", "limitations_analysis", "future_potential"
            ]

        logger.info(f"🔬 Starting comprehensive batch analysis on {len(papers_data)} papers")
        logger.info(f"   Analysis types: {', '.join(analysis_types)}")

        batch_start = datetime.now()
        all_results = []

        for i, paper in enumerate(papers_data):
            logger.info(f"Processing paper {i+1}/{len(papers_data)}: {paper.get('title', 'Unknown')[:60]}...")

            paper_results = await self.analyze_single_paper(paper, analysis_types)
            all_results.extend(paper_results)

            # Add small delay to be respectful to APIs
            await asyncio.sleep(0.5)

        # Compile batch results
        batch_results = self._compile_batch_results(all_results, batch_start)
        self.batch_results = batch_results

        logger.info(f"✅ Completed batch analysis: {len(all_results)} total analyses")
        return batch_results

    async def analyze_single_paper(self, paper: Dict, analysis_types: List[str]) -> List[LLMAnalysisResult]:
        """Analyze a single paper with multiple analysis types"""
        paper_id = paper.get('pmid', f"unknown_{hash(paper.get('title', ''))}")
        title = paper.get('title', 'Unknown Title')

        results = []

        for analysis_type in analysis_types:
            try:
                start_time = datetime.now()

                # Get analysis prompt and run LLM
                prompt = self._build_analysis_prompt(paper, analysis_type)
                response = await self._query_llm(prompt, analysis_type)

                # Extract structured scores from response
                structured_scores = self._extract_structured_scores(response, analysis_type)

                processing_time = (datetime.now() - start_time).total_seconds()

                result = LLMAnalysisResult(
                    paper_id=paper_id,
                    title=title,
                    analysis_type=analysis_type,
                    prompt_used=prompt,
                    response=response,
                    structured_scores=structured_scores,
                    metadata={
                        'journal': paper.get('journal', ''),
                        'year': paper.get('year'),
                        'authors': paper.get('authors', []),
                        'has_fulltext': paper.get('full_text', '') != ''
                    },
                    timestamp=datetime.now().isoformat(),
                    processing_time=processing_time,
                    llm_used=self._get_active_llm()
                )

                results.append(result)
                self.analysis_results.append(result)

            except Exception as e:
                logger.error(f"❌ Failed to analyze {title} with {analysis_type}: {e}")

        return results

    async def comparative_analysis(self, papers_data: List[Dict],
                                 comparison_aspects: List[str] = None) -> Dict[str, Any]:
        """Run comparative analysis across papers"""
        if comparison_aspects is None:
            comparison_aspects = ["methodology_comparison", "findings_synthesis", "quality_ranking"]

        logger.info(f"🔄 Starting comparative analysis on {len(papers_data)} papers")

        comparative_results = {}

        for aspect in comparison_aspects:
            logger.info(f"Analyzing aspect: {aspect}")

            try:
                # Build comparative prompt
                prompt = self._build_comparative_prompt(papers_data, aspect)

                # Query LLM
                response = await self._query_llm(prompt, f"comparative_{aspect}")

                # Structure the response
                comparative_results[aspect] = {
                    "analysis": response,
                    "timestamp": datetime.now().isoformat(),
                    "papers_compared": len(papers_data)
                }

            except Exception as e:
                logger.error(f"❌ Comparative analysis failed for {aspect}: {e}")
                comparative_results[aspect] = {"error": str(e)}

        return {
            "comparative_analysis": comparative_results,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_papers": len(papers_data)
        }

    async def generate_research_insights(self, papers_data: List[Dict]) -> Dict[str, Any]:
        """Generate high-level research insights using LLM"""
        logger.info("💡 Generating research insights")

        insight_prompts = {
            "field_overview": "Provide a comprehensive overview of this research field based on these papers.",
            "emerging_trends": "Identify emerging trends and patterns in this research area.",
            "research_gaps": "Analyze what research gaps and opportunities exist in this field.",
            "methodological_evolution": "How have research methodologies evolved in this field?",
            "future_directions": "What are the most promising future research directions?",
            "cross_disciplinary_connections": "What connections to other research fields are evident?"
        }

        insights = {}

        for insight_type, base_prompt in insight_prompts.items():
            try:
                # Build comprehensive context from all papers
                context = self._build_comprehensive_context(papers_data)

                full_prompt = f"""
                {base_prompt}

                Based on the following collection of {len(papers_data)} research papers:

                {context}

                Please provide a detailed analysis that:
                1. Synthesizes information across all papers
                2. Identifies key patterns and themes
                3. Provides specific examples and evidence
                4. Offers actionable insights for researchers
                """

                response = await self._query_llm(full_prompt, f"insight_{insight_type}")

                insights[insight_type] = {
                    "analysis": response,
                    "timestamp": datetime.now().isoformat(),
                    "papers_analyzed": len(papers_data)
                }

            except Exception as e:
                logger.error(f"❌ Failed to generate insight for {insight_type}: {e}")
                insights[insight_type] = {"error": str(e)}

        return {
            "research_insights": insights,
            "generation_timestamp": datetime.now().isoformat(),
            "corpus_size": len(papers_data)
        }

    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all analyses performed"""
        if not self.analysis_results:
            return {"error": "No analyses have been performed yet"}

        # Basic statistics
        total_analyses = len(self.analysis_results)
        analysis_types = Counter([r.analysis_type for r in self.analysis_results])
        llm_usage = Counter([r.llm_used for r in self.analysis_results])

        # Processing time statistics
        processing_times = [r.processing_time for r in self.analysis_results]
        avg_processing_time = np.mean(processing_times)

        # Score statistics
        all_scores = {}
        for result in self.analysis_results:
            for score_name, score_value in result.structured_scores.items():
                if score_name not in all_scores:
                    all_scores[score_name] = []
                all_scores[score_name].append(score_value)

        score_statistics = {}
        for score_name, scores in all_scores.items():
            if scores:
                score_statistics[score_name] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "count": len(scores)
                }

        return {
            "total_analyses": total_analyses,
            "analysis_types": dict(analysis_types),
            "llm_usage": dict(llm_usage),
            "processing_statistics": {
                "avg_processing_time": avg_processing_time,
                "total_processing_time": sum(processing_times),
                "fastest_analysis": min(processing_times),
                "slowest_analysis": max(processing_times)
            },
            "score_statistics": score_statistics,
            "generation_timestamp": datetime.now().isoformat()
        }

    def get_top_papers_by_score(self, score_type: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get top papers by a specific score type"""
        relevant_results = [r for r in self.analysis_results if score_type in r.structured_scores]

        if not relevant_results:
            return []

        # Sort by score
        sorted_results = sorted(relevant_results,
                              key=lambda x: x.structured_scores[score_type],
                              reverse=True)

        top_papers = []
        for result in sorted_results[:top_n]:
            top_papers.append({
                "title": result.title,
                "paper_id": result.paper_id,
                "score": result.structured_scores[score_type],
                "journal": result.metadata.get("journal", ""),
                "year": result.metadata.get("year"),
                "analysis_type": result.analysis_type
            })

        return top_papers

    async def _query_llm(self, prompt: str, analysis_type: str) -> str:
        """Query available LLM with error handling"""
        if self.openai_key:
            try:
                return await self._query_openai(prompt, analysis_type)
            except Exception as e:
                logger.warning(f"OpenAI query failed: {e}")

        if self.deepseek_key:
            try:
                return await self._query_deepseek(prompt, analysis_type)
            except Exception as e:
                logger.warning(f"DeepSeek query failed: {e}")

        # Fallback to basic response
        return f"LLM analysis unavailable. Analysis type: {analysis_type}"

    async def _query_openai(self, prompt: str, analysis_type: str) -> str:
        """Query OpenAI API"""
        client = openai.OpenAI(api_key=self.openai_key)

        # Determine model based on analysis complexity
        model = "gpt-4" if "comparative" in analysis_type or "insight" in analysis_type else "gpt-3.5-turbo"

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert scientific literature analyst and researcher. Provide detailed, evidence-based analyses."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800 if "comparative" in analysis_type else 500,
            temperature=0.2
        )

        return response.choices[0].message.content

    async def _query_deepseek(self, prompt: str, analysis_type: str) -> str:
        """Query DeepSeek API"""
        # DeepSeek API implementation would go here
        # For now, return placeholder
        return f"DeepSeek analysis for {analysis_type} would be implemented here"

    def _get_active_llm(self) -> str:
        """Get the active LLM being used"""
        if self.openai_key:
            return "openai"
        elif self.deepseek_key:
            return "deepseek"
        else:
            return "none"

    def _load_analysis_templates(self) -> Dict[str, str]:
        """Load analysis prompt templates"""
        return {
            "scientific_quality": """
            Analyze the scientific quality of this research paper. Consider:
            1. Research design and methodology rigor
            2. Statistical analysis appropriateness
            3. Sample size and data quality
            4. Reproducibility of methods
            5. Clarity of objectives and hypotheses

            Paper: "{title}"
            Abstract: {abstract}
            Content: {content}

            Provide scores (1-10) for: methodology_rigor, statistical_quality, reproducibility, clarity, overall_quality
            Format: [SCORE:methodology_rigor:X] [SCORE:statistical_quality:X] etc.
            """,

            "methodology_assessment": """
            Assess the methodological approach of this research. Analyze:
            1. Appropriateness of methods for research questions
            2. Innovation in methodology
            3. Potential limitations and biases
            4. Comparison with standard practices
            5. Technical execution quality

            Paper: "{title}"
            Abstract: {abstract}
            Content: {content}

            Provide scores (1-10) for: method_appropriateness, innovation, bias_control, technical_execution, overall_methodology
            Format: [SCORE:method_appropriateness:X] [SCORE:innovation:X] etc.
            """,

            "innovation_score": """
            Evaluate the innovation and novelty of this research. Consider:
            1. Novelty of research questions or approaches
            2. Original contributions to the field
            3. Creative methodological innovations
            4. Breakthrough potential
            5. Paradigm-shifting implications

            Paper: "{title}"
            Abstract: {abstract}
            Content: {content}

            Provide scores (1-10) for: novelty, originality, creativity, breakthrough_potential, paradigm_impact
            Format: [SCORE:novelty:X] [SCORE:originality:X] etc.
            """,

            "research_impact": """
            Assess the potential research impact of this work. Evaluate:
            1. Significance for the field
            2. Practical applications potential
            3. Influence on future research directions
            4. Societal or clinical relevance
            5. Cross-disciplinary impact potential

            Paper: "{title}"
            Abstract: {abstract}
            Content: {content}

            Provide scores (1-10) for: field_significance, practical_applications, future_influence, societal_relevance, cross_disciplinary
            Format: [SCORE:field_significance:X] [SCORE:practical_applications:X] etc.
            """,

            "limitations_analysis": """
            Critically analyze the limitations of this research. Identify:
            1. Methodological limitations
            2. Sample or data limitations
            3. Analytical constraints
            4. Generalizability issues
            5. Potential confounding factors

            Paper: "{title}"
            Abstract: {abstract}
            Content: {content}

            Provide scores (1-10, where 10 = minimal limitations) for: method_limitations, data_limitations, analytical_constraints, generalizability, confounding_control
            Format: [SCORE:method_limitations:X] [SCORE:data_limitations:X] etc.
            """,

            "future_potential": """
            Evaluate the future research potential of this work. Consider:
            1. Follow-up research opportunities
            2. Scalability of findings
            3. Technology transfer potential
            4. Long-term research trajectory
            5. Interdisciplinary expansion possibilities

            Paper: "{title}"
            Abstract: {abstract}
            Content: {content}

            Provide scores (1-10) for: followup_opportunities, scalability, tech_transfer, longterm_trajectory, interdisciplinary_potential
            Format: [SCORE:followup_opportunities:X] [SCORE:scalability:X] etc.
            """
        }

    def _build_analysis_prompt(self, paper: Dict, analysis_type: str) -> str:
        """Build analysis prompt for a specific paper and analysis type"""
        template = self.analysis_templates.get(analysis_type, self.analysis_templates["scientific_quality"])

        # Extract paper content
        title = paper.get('title', 'Unknown Title')
        abstract = paper.get('abstract', 'No abstract available')

        # Use full text if available, otherwise use abstract
        content = paper.get('full_text', '')
        if not content or len(content.strip()) < 100:
            content = abstract

        # Truncate content if too long
        if len(content) > 3000:
            content = content[:3000] + "... [truncated]"

        return template.format(
            title=title,
            abstract=abstract,
            content=content
        )

    def _build_comparative_prompt(self, papers_data: List[Dict], aspect: str) -> str:
        """Build comparative analysis prompt"""
        paper_summaries = []

        for i, paper in enumerate(papers_data[:10]):  # Limit to 10 papers for comparison
            title = paper.get('title', 'Unknown Title')
            abstract = paper.get('abstract', 'No abstract available')

            paper_summaries.append(f"""
            Paper {i+1}: {title}
            Abstract: {abstract[:300]}{'...' if len(abstract) > 300 else ''}
            """)

        papers_text = "\n".join(paper_summaries)

        prompts = {
            "methodology_comparison": f"""
            Compare and contrast the methodological approaches used in these {len(papers_data)} research papers:

            {papers_text}

            Analyze:
            1. Common methodological patterns
            2. Innovative or unique approaches
            3. Methodological strengths and weaknesses
            4. Evolution of methods across studies
            5. Recommendations for future methodological improvements
            """,

            "findings_synthesis": f"""
            Synthesize the key findings from these {len(papers_data)} research papers:

            {papers_text}

            Provide:
            1. Convergent findings across studies
            2. Conflicting or contradictory results
            3. Novel insights unique to specific studies
            4. Gaps in current understanding
            5. Integrated conclusions from the body of work
            """,

            "quality_ranking": f"""
            Rank and compare the scientific quality of these {len(papers_data)} research papers:

            {papers_text}

            Evaluate and rank based on:
            1. Methodological rigor
            2. Statistical analysis quality
            3. Research design appropriateness
            4. Contribution to knowledge
            5. Overall scientific merit

            Provide a ranked list with justification for rankings.
            """
        }

        return prompts.get(aspect, prompts["methodology_comparison"])

    def _build_comprehensive_context(self, papers_data: List[Dict], max_papers: int = 20) -> str:
        """Build comprehensive context from multiple papers"""
        context_parts = []

        for i, paper in enumerate(papers_data[:max_papers]):
            title = paper.get('title', 'Unknown Title')
            abstract = paper.get('abstract', 'No abstract available')
            year = paper.get('year', 'Unknown year')
            journal = paper.get('journal', 'Unknown journal')

            context_parts.append(f"""
            Paper {i+1} ({year}): {title}
            Journal: {journal}
            Abstract: {abstract[:200]}{'...' if len(abstract) > 200 else ''}
            """)

        if len(papers_data) > max_papers:
            context_parts.append(f"\n[Note: {len(papers_data) - max_papers} additional papers not shown for brevity]")

        return "\n".join(context_parts)

    def _extract_structured_scores(self, response: str, analysis_type: str) -> Dict[str, float]:
        """Extract structured scores from LLM response"""
        scores = {}

        # Look for score patterns like [SCORE:metric_name:8.5]
        import re
        score_pattern = r'\[SCORE:([^:]+):([0-9.]+)\]'
        matches = re.findall(score_pattern, response)

        for metric_name, score_str in matches:
            try:
                scores[metric_name] = float(score_str)
            except ValueError:
                logger.warning(f"Could not parse score: {metric_name}={score_str}")

        # If no structured scores found, try to extract from text
        if not scores:
            scores = self._extract_scores_from_text(response, analysis_type)

        return scores

    def _extract_scores_from_text(self, response: str, analysis_type: str) -> Dict[str, float]:
        """Extract scores from unstructured text as fallback"""
        # This is a simple fallback - could be made more sophisticated
        import re

        # Look for patterns like "8/10", "7.5 out of 10", etc.
        score_patterns = [
            r'(\d+\.?\d*)/10',
            r'(\d+\.?\d*)\s*out\s*of\s*10',
            r'score.*?(\d+\.?\d*)',
            r'rating.*?(\d+\.?\d*)'
        ]

        found_scores = []
        for pattern in score_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            found_scores.extend([float(m) for m in matches if float(m) <= 10])

        if found_scores:
            # Assign a generic overall score
            avg_score = np.mean(found_scores)
            return {"overall_score": avg_score}

        return {"overall_score": 5.0}  # Default neutral score

    def _compile_batch_results(self, all_results: List[LLMAnalysisResult], batch_start: datetime) -> Dict[str, Any]:
        """Compile comprehensive batch analysis results"""
        batch_end = datetime.now()
        total_time = (batch_end - batch_start).total_seconds()

        # Group results by analysis type
        results_by_type = defaultdict(list)
        for result in all_results:
            results_by_type[result.analysis_type].append(result)

        # Calculate aggregate statistics
        analysis_summary = {}
        for analysis_type, results in results_by_type.items():
            scores = defaultdict(list)
            for result in results:
                for score_name, score_value in result.structured_scores.items():
                    scores[score_name].append(score_value)

            # Calculate statistics for each score type
            score_stats = {}
            for score_name, score_values in scores.items():
                if score_values:
                    score_stats[score_name] = {
                        "mean": float(np.mean(score_values)),
                        "std": float(np.std(score_values)),
                        "min": float(np.min(score_values)),
                        "max": float(np.max(score_values)),
                        "median": float(np.median(score_values))
                    }

            analysis_summary[analysis_type] = {
                "total_analyses": len(results),
                "score_statistics": score_stats,
                "avg_processing_time": np.mean([r.processing_time for r in results])
            }

        return {
            "batch_analysis_results": {
                "summary": analysis_summary,
                "total_analyses": len(all_results),
                "unique_papers": len(set(r.paper_id for r in all_results)),
                "analysis_types": list(results_by_type.keys()),
                "batch_processing_time": total_time,
                "avg_analysis_time": np.mean([r.processing_time for r in all_results]),
                "batch_timestamp": batch_end.isoformat()
            },
            "detailed_results": [
                {
                    "paper_id": r.paper_id,
                    "title": r.title,
                    "analysis_type": r.analysis_type,
                    "response": r.response,
                    "structured_scores": r.structured_scores,
                    "metadata": r.metadata,
                    "processing_time": r.processing_time,
                    "llm_used": r.llm_used
                }
                for r in all_results
            ]
        }

    def save_results(self, filename: str = None) -> str:
        """Save all analysis results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"llm_analysis_results_{timestamp}.json"

        output = {
            "analysis_statistics": self.get_analysis_statistics(),
            "batch_results": self.batch_results,
            "all_results": [
                {
                    "paper_id": r.paper_id,
                    "title": r.title,
                    "analysis_type": r.analysis_type,
                    "response": r.response,
                    "structured_scores": r.structured_scores,
                    "metadata": r.metadata,
                    "timestamp": r.timestamp,
                    "processing_time": r.processing_time,
                    "llm_used": r.llm_used
                }
                for r in self.analysis_results
            ],
            "export_timestamp": datetime.now().isoformat()
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"💾 Saved LLM analysis results to {filename}")
        return filename