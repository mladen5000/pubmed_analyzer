#!/usr/bin/env python3
"""
Enhanced PubMed Literature Analysis Pipeline
Advanced NLP/ML analysis with RAG second-phase capabilities
"""

import os
import sys
import asyncio
import argparse
import logging
import json
from typing import List, Optional, Dict, Any
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import components
from pubmed_analyzer import (
    Paper,
    PubMedSearcher,
    PMIDToPMCConverter,
    UnifiedPDFFetcher
)

# Import new advanced modules
from pubmed_analyzer.core.nlp_analyzer import AdvancedNLPAnalyzer
from pubmed_analyzer.core.rag_analyzer import EnhancedRAGAnalyzer


class EnhancedPubMedPipeline:
    """Enhanced PubMed analysis pipeline with advanced NLP/ML and RAG capabilities"""

    def __init__(
        self,
        email: str,
        api_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        deepseek_key: Optional[str] = None,
    ):
        self.email = email
        self.api_key = api_key
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        self.deepseek_key = deepseek_key or os.getenv("DEEPSEEK_API_KEY")

        # Initialize core components
        self.searcher = PubMedSearcher(email, api_key)
        self.id_converter = PMIDToPMCConverter(email, api_key)
        self.pdf_fetcher = UnifiedPDFFetcher()

        # Initialize advanced analysis components
        self.nlp_analyzer = AdvancedNLPAnalyzer()
        self.rag_analyzer = EnhancedRAGAnalyzer(openai_key, deepseek_key)

        logger.info("🚀 Enhanced PubMed Pipeline initialized")
        logger.info(f"   Email: {email}")
        if api_key:
            logger.info("   NCBI API key: ✅")
        if self.openai_key:
            logger.info("   OpenAI API key: ✅")
        if self.deepseek_key:
            logger.info("   DeepSeek API key: ✅")

    async def run_comprehensive_analysis(
        self,
        query: str,
        max_papers: int = 50,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        enable_rag: bool = True,
        custom_questions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run comprehensive analysis pipeline with all features"""

        analysis_start = datetime.now()
        logger.info("🔬 Starting Enhanced PubMed Analysis Pipeline")
        logger.info(f"   Query: '{query}'")
        logger.info(f"   Max papers: {max_papers}")
        logger.info(f"   RAG enabled: {enable_rag}")

        results = {
            "pipeline_info": {
                "query": query,
                "max_papers": max_papers,
                "start_date": start_date,
                "end_date": end_date,
                "analysis_timestamp": analysis_start.isoformat(),
                "rag_enabled": enable_rag,
            },
            "phases": {}
        }

        try:
            # Phase 1: Data Collection
            logger.info("\n📚 PHASE 1: DATA COLLECTION")
            papers = await self._data_collection_phase(query, max_papers, start_date, end_date)

            papers_data = [self._paper_to_dict(paper) for paper in papers]
            results["phases"]["data_collection"] = {
                "total_papers": len(papers_data),
                "successful_downloads": sum(1 for p in papers if p.download_success),
                "completion_time": (datetime.now() - analysis_start).total_seconds()
            }

            if not papers_data:
                logger.warning("❌ No papers collected. Stopping analysis.")
                return results

            # Phase 2: Advanced NLP Analysis
            logger.info("\n🧠 PHASE 2: ADVANCED NLP ANALYSIS")
            phase2_start = datetime.now()
            nlp_results = await self._advanced_nlp_phase(papers_data)
            results["phases"]["nlp_analysis"] = {
                **nlp_results,
                "completion_time": (datetime.now() - phase2_start).total_seconds()
            }

            # Phase 3: RAG Analysis (if enabled)
            if enable_rag and (self.openai_key or self.deepseek_key):
                logger.info("\n🤖 PHASE 3: RAG-POWERED ANALYSIS")
                phase3_start = datetime.now()
                rag_results = await self._rag_analysis_phase(papers_data, custom_questions)
                results["phases"]["rag_analysis"] = {
                    **rag_results,
                    "completion_time": (datetime.now() - phase3_start).total_seconds()
                }

            # Phase 4: Generate Comprehensive Report
            logger.info("\n📊 PHASE 4: COMPREHENSIVE REPORTING")
            phase4_start = datetime.now()
            report = self._generate_comprehensive_report(results, papers_data)
            results["comprehensive_report"] = report
            results["phases"]["reporting"] = {
                "completion_time": (datetime.now() - phase4_start).total_seconds()
            }

            # Save results
            await self._save_results(results, query)

            total_time = (datetime.now() - analysis_start).total_seconds()
            logger.info(f"✅ Analysis complete in {total_time:.1f} seconds")
            results["pipeline_info"]["total_analysis_time"] = total_time

            return results

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            results["error"] = {
                "message": str(e),
                "type": type(e).__name__,
                "timestamp": datetime.now().isoformat()
            }
            return results

    async def _data_collection_phase(
        self, query: str, max_papers: int, start_date: Optional[str], end_date: Optional[str]
    ) -> List[Paper]:
        """Phase 1: Enhanced data collection with robust error handling"""

        # Search for papers
        logger.info("🔍 Searching PubMed...")
        pmids = await self.searcher.search_papers(
            query=query,
            max_results=max_papers,
            start_date=start_date,
            end_date=end_date
        )

        if not pmids:
            logger.warning("❌ No papers found")
            return []

        logger.info(f"✅ Found {len(pmids)} papers")

        # Get paper metadata
        logger.info("📝 Collecting paper metadata...")
        papers = await self.searcher.fetch_papers_metadata(pmids)
        logger.info(f"✅ Collected metadata for {len(papers)} papers")

        # Convert to PMC IDs for full-text access
        logger.info("🔓 Converting to PMC IDs...")
        await self.id_converter.enrich_with_pmcids(papers)

        pmcid_count = sum(1 for p in papers if p.pmcid)
        logger.info(f"✅ Found PMC IDs for {pmcid_count} papers")

        # Download PDFs with enhanced error handling
        logger.info("📄 Downloading PDFs...")
        await self.pdf_fetcher.download_all(papers)

        successful_downloads = sum(1 for p in papers if p.download_success)
        logger.info(f"✅ Successfully downloaded {successful_downloads} PDFs")

        return papers

    async def _advanced_nlp_phase(self, papers_data: List[Dict]) -> Dict[str, Any]:
        """Phase 2: Advanced NLP analysis"""

        nlp_results = {
            "timestamp": datetime.now().isoformat(),
            "analyses_performed": []
        }

        # Extract texts for analysis
        abstracts = [p.get('abstract', '') for p in papers_data if p.get('abstract')]
        full_texts = [p.get('full_text', '') for p in papers_data if p.get('full_text')]
        all_texts = abstracts + full_texts

        if not all_texts:
            logger.warning("❌ No text data available for NLP analysis")
            return {"error": "No text data available"}

        try:
            # 1. Advanced Topic Modeling
            logger.info("🎯 Advanced topic modeling...")
            topics = self.nlp_analyzer.advanced_topic_modeling(all_texts, n_topics=8)
            nlp_results["topic_modeling"] = topics
            nlp_results["analyses_performed"].append("topic_modeling")

            # 2. Enhanced Sentiment Analysis
            logger.info("😊 Enhanced sentiment analysis...")
            sentiment = self.nlp_analyzer.enhanced_sentiment_analysis(all_texts)
            nlp_results["sentiment_analysis"] = sentiment
            nlp_results["analyses_performed"].append("sentiment_analysis")

            # 3. Advanced Clustering
            logger.info("🎯 Advanced document clustering...")
            clusters = self.nlp_analyzer.advanced_clustering(abstracts[:50], method="kmeans")
            nlp_results["clustering"] = clusters
            nlp_results["analyses_performed"].append("clustering")

            # 4. Named Entity Recognition
            logger.info("🔍 Named entity recognition...")
            entities = self.nlp_analyzer.named_entity_recognition(abstracts[:30])
            nlp_results["named_entities"] = entities
            nlp_results["analyses_performed"].append("named_entity_recognition")

            # 5. Research Trend Analysis
            logger.info("📈 Research trend analysis...")
            trends = self.nlp_analyzer.research_trend_analysis(papers_data)
            nlp_results["trend_analysis"] = trends
            nlp_results["analyses_performed"].append("trend_analysis")

            # 6. Citation Network Analysis
            logger.info("🕸️ Citation network analysis...")
            network = self.nlp_analyzer.citation_network_analysis(papers_data)
            nlp_results["network_analysis"] = network
            nlp_results["analyses_performed"].append("network_analysis")

            # 7. Research Gap Identification
            logger.info("🔍 Research gap identification...")
            gaps = self.nlp_analyzer.research_gap_identification(all_texts, topics)
            nlp_results["research_gaps"] = gaps
            nlp_results["analyses_performed"].append("research_gap_identification")

            logger.info(f"✅ Completed {len(nlp_results['analyses_performed'])} NLP analyses")

        except Exception as e:
            logger.error(f"❌ NLP analysis failed: {e}")
            nlp_results["error"] = str(e)

        return nlp_results

    async def _rag_analysis_phase(self, papers_data: List[Dict], custom_questions: Optional[List[str]]) -> Dict[str, Any]:
        """Phase 3: RAG-powered analysis"""

        rag_results = {
            "timestamp": datetime.now().isoformat(),
            "rag_components": []
        }

        try:
            # 1. Build Vector Indices
            logger.info("🔍 Building vector indices for RAG...")
            indices_info = self.rag_analyzer.build_vector_indices(papers_data)
            rag_results["vector_indices"] = indices_info
            rag_results["rag_components"].append("vector_indices")

            # 2. Interactive Analysis Session
            logger.info("🎯 Running interactive analysis session...")
            interactive_results = self.rag_analyzer.interactive_analysis_session(papers_data)
            rag_results["interactive_session"] = interactive_results
            rag_results["rag_components"].append("interactive_session")

            # 3. Generate Research Insights
            logger.info("💡 Generating research insights...")
            insights = self.rag_analyzer.generate_research_insights(papers_data)
            rag_results["research_insights"] = insights
            rag_results["rag_components"].append("research_insights")

            # 4. Custom Questions Analysis
            if custom_questions:
                logger.info(f"📝 Analyzing {len(custom_questions)} custom questions...")
                custom_results = self.rag_analyzer.custom_query_analysis(custom_questions)
                rag_results["custom_questions"] = custom_results
                rag_results["rag_components"].append("custom_questions")

            logger.info(f"✅ Completed {len(rag_results['rag_components'])} RAG analyses")

        except Exception as e:
            logger.error(f"❌ RAG analysis failed: {e}")
            rag_results["error"] = str(e)

        return rag_results

    def _generate_comprehensive_report(self, results: Dict[str, Any], papers_data: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""

        report = {
            "generation_timestamp": datetime.now().isoformat(),
            "executive_summary": {},
            "detailed_findings": {},
            "recommendations": []
        }

        # Executive Summary
        total_papers = len(papers_data)

        # NLP Analysis Summary
        nlp_phase = results.get("phases", {}).get("nlp_analysis", {})
        nlp_analyses = len(nlp_phase.get("analyses_performed", []))

        # RAG Analysis Summary
        rag_phase = results.get("phases", {}).get("rag_analysis", {})
        rag_components = len(rag_phase.get("rag_components", []))

        report["executive_summary"] = {
            "corpus_size": total_papers,
            "nlp_analyses_performed": nlp_analyses,
            "rag_components_analyzed": rag_components,
            "total_analysis_time": results.get("pipeline_info", {}).get("total_analysis_time", 0),
            "key_insights": self._extract_key_insights(results)
        }

        # Detailed Findings
        report["detailed_findings"] = {
            "top_topics": self._extract_top_topics(nlp_phase),
            "sentiment_overview": self._extract_sentiment_overview(nlp_phase),
            "research_gaps": self._extract_research_gaps(nlp_phase, rag_phase),
            "trend_analysis": self._extract_trend_summary(nlp_phase)
        }

        # Recommendations
        report["recommendations"] = self._generate_recommendations(results)

        return report

    async def _save_results(self, results: Dict[str, Any], query: str):
        """Save analysis results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_slug = query.replace(" ", "_").replace("/", "_")[:20]

        # Save comprehensive results
        filename = f"enhanced_analysis_{query_slug}_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"📁 Saved comprehensive results to {filename}")

        # Save executive report
        if "comprehensive_report" in results:
            report_filename = f"executive_report_{query_slug}_{timestamp}.json"
            with open(report_filename, 'w') as f:
                json.dump(results["comprehensive_report"], f, indent=2, default=str)
            logger.info(f"📊 Saved executive report to {report_filename}")

    def _paper_to_dict(self, paper: Paper) -> Dict[str, Any]:
        """Convert Paper object to dictionary with safe attribute access"""
        return {
            "pmid": getattr(paper, 'pmid', ''),
            "pmcid": getattr(paper, 'pmcid', None),
            "title": getattr(paper, 'title', ''),
            "abstract": getattr(paper, 'abstract', ''),
            "authors": getattr(paper, 'authors', []),
            "journal": getattr(paper, 'journal', ''),
            "year": getattr(paper, 'year', None),
            "keywords": getattr(paper, 'keywords', []),
            "full_text": getattr(paper, 'full_text', ''),
            "sections": getattr(paper, 'sections', {}),
            "download_success": getattr(paper, 'download_success', False),
            "has_fulltext": getattr(paper, 'has_fulltext', False)
        }

    def _extract_key_insights(self, results: Dict[str, Any]) -> List[str]:
        """Extract key insights from analysis results"""
        insights = []

        # From NLP analysis
        nlp_results = results.get("phases", {}).get("nlp_analysis", {})

        # Topic modeling insights
        topics = nlp_results.get("topic_modeling", {}).get("topics", {})
        if topics:
            top_topic = max(topics.items(), key=lambda x: x[1].get("coherence", 0))
            top_words = [word for word, _ in top_topic[1].get("words", [])[:3]]
            insights.append(f"Primary research theme involves: {', '.join(top_words)}")

        # Sentiment insights
        sentiment = nlp_results.get("sentiment_analysis", {}).get("overall_sentiment", {})
        if sentiment:
            polarity = sentiment.get("mean_polarity", 0)
            if polarity > 0.1:
                insights.append("Overall sentiment in literature is positive")
            elif polarity < -0.1:
                insights.append("Overall sentiment in literature shows concerns/challenges")

        # Research gaps
        gaps = nlp_results.get("research_gaps", [])
        if gaps:
            insights.append(f"Identified {len(gaps)} potential research gaps")

        return insights[:5]  # Top 5 insights

    def _extract_top_topics(self, nlp_results: Dict) -> List[Dict]:
        """Extract top topics from NLP analysis"""
        topics_data = nlp_results.get("topic_modeling", {}).get("topics", {})

        topics_list = []
        for topic_id, topic_info in topics_data.items():
            topics_list.append({
                "topic_id": topic_id,
                "coherence": topic_info.get("coherence", 0),
                "top_words": [word for word, _ in topic_info.get("words", [])[:5]]
            })

        return sorted(topics_list, key=lambda x: x["coherence"], reverse=True)[:5]

    def _extract_sentiment_overview(self, nlp_results: Dict) -> Dict:
        """Extract sentiment analysis overview"""
        sentiment_data = nlp_results.get("sentiment_analysis", {}).get("overall_sentiment", {})
        return {
            "mean_polarity": sentiment_data.get("mean_polarity", 0),
            "positive_ratio": sentiment_data.get("positive_ratio", 0),
            "negative_ratio": sentiment_data.get("negative_ratio", 0),
            "neutral_ratio": sentiment_data.get("neutral_ratio", 0)
        }

    def _extract_research_gaps(self, nlp_results: Dict, rag_results: Dict) -> List[Dict]:
        """Extract identified research gaps"""
        gaps = nlp_results.get("research_gaps", [])

        # Add RAG-identified gaps if available
        rag_insights = rag_results.get("research_insights", {}).get("insights", {})
        gap_insight = rag_insights.get("research_gaps", {})

        return gaps[:5]  # Top 5 gaps

    def _extract_trend_summary(self, nlp_results: Dict) -> Dict:
        """Extract trend analysis summary"""
        trends = nlp_results.get("trend_analysis", {})
        return {
            "publication_trends": trends.get("publication_trends", {}),
            "journal_trends": trends.get("journal_trends", {}),
            "keyword_evolution": bool(trends.get("keyword_evolution"))
        }

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        nlp_results = results.get("phases", {}).get("nlp_analysis", {})

        # Based on research gaps
        gaps = nlp_results.get("research_gaps", [])
        if gaps:
            for gap in gaps[:2]:
                if gap.get("type") == "methodology_gap":
                    recommendations.append("Consider expanding methodological diversity in future research")
                elif gap.get("type") == "weak_topic_coverage":
                    keywords = gap.get("keywords", [])
                    recommendations.append(f"Explore underresearched areas: {', '.join(keywords[:2])}")

        # Based on trends
        trends = nlp_results.get("trend_analysis", {})
        pub_trends = trends.get("publication_trends", {})
        if pub_trends.get("trend_slope", 0) > 0:
            recommendations.append("Field shows positive growth trend - consider staying current with developments")

        # Based on sentiment
        sentiment = nlp_results.get("sentiment_analysis", {}).get("overall_sentiment", {})
        if sentiment.get("negative_ratio", 0) > 0.3:
            recommendations.append("Address commonly mentioned limitations and challenges")

        return recommendations[:5]


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Enhanced PubMed Literature Analysis Pipeline with NLP/ML and RAG"
    )

    parser.add_argument(
        "--query", "-q", type=str, required=True, help="PubMed search query"
    )
    parser.add_argument(
        "--max-papers", "-m", type=int, default=50, help="Maximum papers to analyze"
    )
    parser.add_argument(
        "--email", "-e", type=str, required=True, help="Email for NCBI API access"
    )
    parser.add_argument(
        "--api-key", "-k", type=str, help="NCBI API key (optional, for higher rate limits)"
    )
    parser.add_argument(
        "--openai-key", type=str, help="OpenAI API key for RAG analysis"
    )
    parser.add_argument(
        "--deepseek-key", type=str, help="DeepSeek API key for RAG analysis"
    )
    parser.add_argument(
        "--start-date", type=str, help="Start date filter (YYYY/MM/DD)"
    )
    parser.add_argument(
        "--end-date", type=str, help="End date filter (YYYY/MM/DD)"
    )
    parser.add_argument(
        "--disable-rag", action="store_true", help="Disable RAG analysis phase"
    )
    parser.add_argument(
        "--custom-questions", nargs="+", help="Custom questions for RAG analysis"
    )

    return parser.parse_args()


async def main():
    """Main execution function"""
    args = parse_arguments()

    logger.info("🚀 Enhanced PubMed Analysis Pipeline Starting...")
    logger.info("=" * 60)

    # Initialize pipeline
    pipeline = EnhancedPubMedPipeline(
        email=args.email,
        api_key=args.api_key,
        openai_key=args.openai_key,
        deepseek_key=args.deepseek_key,
    )

    # Run comprehensive analysis
    try:
        results = await pipeline.run_comprehensive_analysis(
            query=args.query,
            max_papers=args.max_papers,
            start_date=args.start_date,
            end_date=args.end_date,
            enable_rag=not args.disable_rag,
            custom_questions=args.custom_questions,
        )

        # Print summary
        if "error" not in results:
            logger.info("\n" + "=" * 60)
            logger.info("📊 ANALYSIS SUMMARY")
            logger.info("=" * 60)

            summary = results.get("comprehensive_report", {}).get("executive_summary", {})
            logger.info(f"📚 Papers analyzed: {summary.get('corpus_size', 0)}")
            logger.info(f"🧠 NLP analyses: {summary.get('nlp_analyses_performed', 0)}")
            logger.info(f"🤖 RAG components: {summary.get('rag_components_analyzed', 0)}")
            logger.info(f"⏱️  Total time: {summary.get('total_analysis_time', 0):.1f}s")

            insights = summary.get("key_insights", [])
            if insights:
                logger.info("\n💡 Key Insights:")
                for insight in insights:
                    logger.info(f"   • {insight}")

        else:
            logger.error(f"❌ Analysis failed: {results['error']['message']}")

    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())