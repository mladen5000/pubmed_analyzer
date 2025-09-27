#!/usr/bin/env python3
"""
Modular Scientific Literature Analysis Pipeline
Combines full-text PDF processing, RAG, vector search, and advanced analytics
"""

import os
import sys
import asyncio
import argparse
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import our modular components
from pubmed_analyzer.models.paper import Paper
from pubmed_analyzer.core.search import PubMedSearcher
from pubmed_analyzer.core.id_converter import PMIDToPMCConverter
from pubmed_analyzer.api.pdf_fetcher_api import PubMedPDFFetcher

# Import enhanced analysis components
try:
    from pubmed_analyzer.core.llm_analyzer import ComprehensiveLLMAnalyzer
    from pubmed_analyzer.utils.visualizer import EnhancedVisualizer
    from pubmed_analyzer.utils.abstract_visualizer import AbstractOptimizedVisualizer
    LLM_AVAILABLE = True
except ImportError:
    logger.warning("Enhanced LLM analysis components not available")
    LLM_AVAILABLE = False


class ModularPubMedPipeline:
    """Modular PubMed analysis pipeline with unified PDF fetching"""

    def __init__(
        self,
        email: str,
        api_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        deepseek_key: Optional[str] = None,
        skip_pdf_download: bool = True,
    ):
        self.email = email
        self.api_key = api_key
        self.openai_key = openai_key
        self.deepseek_key = deepseek_key
        self.skip_pdf_download = skip_pdf_download

        # Initialize core components
        self.searcher = PubMedSearcher(email, api_key)

        # Only initialize PDF-related components if needed
        if not skip_pdf_download:
            self.id_converter = PMIDToPMCConverter(email, api_key)
            self.pdf_fetcher = PubMedPDFFetcher(
                email=email,
                api_key=api_key,
                enhanced_mode=True,  # Enable third-party sources for higher success rates
                batch_size=5,
                min_success_rate=0.3
            )
        else:
            self.id_converter = None
            self.pdf_fetcher = None

        # Initialize enhanced analysis components
        if LLM_AVAILABLE:
            self.llm_analyzer = ComprehensiveLLMAnalyzer(openai_key, deepseek_key)
            self.visualizer = EnhancedVisualizer()
            self.abstract_visualizer = AbstractOptimizedVisualizer()
        else:
            self.llm_analyzer = None
            self.visualizer = None
            self.abstract_visualizer = None

        logger.info(f"Initialized modular pipeline with email: {email}")
        if api_key:
            logger.info("Using NCBI API key - higher rate limits enabled")

    async def run_pipeline(
        self,
        query: str,
        max_papers: int = 50,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        enable_llm_analysis: bool = False,
        enable_visualizations: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline

        Args:
            query: PubMed search query
            max_papers: Maximum number of papers to analyze
            start_date: Start date filter (YYYY/MM/DD)
            end_date: End date filter (YYYY/MM/DD)
            enable_llm_analysis: Enable LLM-based paper analysis
            enable_visualizations: Enable visualization generation

        Returns:
            Dictionary containing papers, analysis results, and visualizations
        """
        logger.info(f"🔍 Starting pipeline for query: '{query}'")

        # Step 1: Search PubMed
        logger.info("Step 1: Searching PubMed...")
        pmids = await self.searcher.search_papers(
            query=query,
            max_results=max_papers,
            start_date=start_date,
            end_date=end_date,
        )

        if not pmids:
            logger.warning("No papers found for the given query")
            return []

        # Step 2: Fetch metadata
        logger.info("Step 2: Fetching paper metadata...")
        papers = await self.searcher.fetch_papers_metadata(pmids)

        # Step 3: Full-text processing (only if needed)
        if not self.skip_pdf_download:
            logger.info("Step 3: Converting PMIDs to PMC IDs and checking Open Access...")
            await self.id_converter.enrich_with_pmcids(papers)

            # Count papers with full-text potential
            fulltext_papers = [p for p in papers if p.has_fulltext]
            logger.info(
                f"Found {len(fulltext_papers)}/{len(papers)} papers with potential full-text access"
            )

            # Step 4: Download PDFs
            logger.info("Step 4: Downloading PDFs using enhanced fetcher...")
            pmids = [p.pmid for p in papers]
            result = await self.pdf_fetcher.download_from_pmids(pmids)
        else:
            logger.info("Step 3-4: ⚡ PURE ABSTRACT MODE - No PMC conversion, no downloads needed!")
            logger.info("📄 Ready for immediate analysis with abstracts")

        # Report final statistics
        if not self.skip_pdf_download:
            successful_downloads = sum(1 for p in papers if p.download_success)
            logger.info(
                f"✅ Core pipeline complete: {successful_downloads} PDFs downloaded successfully"
            )
        else:
            logger.info(
                f"✅ Abstract-only pipeline complete: {len(papers)} abstracts processed"
            )

        # Prepare results dictionary
        results = {
            "papers": papers,
            "query": query,
            "pipeline_summary": self.get_analysis_summary(papers),
            "llm_analysis": None,
            "visualizations": []
        }

        # Step 5: Enhanced LLM Analysis (optional)
        if enable_llm_analysis and self.llm_analyzer and (self.openai_key or self.deepseek_key):
            logger.info("Step 5: Running enhanced LLM analysis...")

            try:
                # Convert papers to format expected by LLM analyzer
                papers_data = []
                for paper in papers:
                    paper_data = {
                        'pmid': paper.pmid,
                        'title': paper.title,
                        'abstract': paper.abstract,
                        'authors': getattr(paper, 'authors', []),
                        'journal': paper.journal,
                        'year': getattr(paper, 'publication_date', ''),
                        'full_text': paper.full_text or '',
                        'sections': getattr(paper, 'sections', {}),
                        'keywords': getattr(paper, 'keywords', [])
                    }
                    papers_data.append(paper_data)

                # Run comprehensive LLM analysis
                llm_results = await self.llm_analyzer.comprehensive_batch_analysis(papers_data)

                # Generate research insights
                insights = await self.llm_analyzer.generate_research_insights(papers_data)
                llm_results.update(insights)

                results["llm_analysis"] = llm_results
                logger.info("✅ LLM analysis complete")

            except Exception as e:
                logger.error(f"❌ LLM analysis failed: {e}")
                results["llm_analysis"] = {"error": str(e)}

        elif enable_llm_analysis:
            logger.warning("⚠️ LLM analysis requested but no API keys available")

        # Step 6: Generate Visualizations (optional)
        if enable_visualizations and (self.visualizer or self.abstract_visualizer):
            logger.info("Step 6: Generating visualizations...")

            try:
                visualization_files = []

                # Abstract-optimized visualizations (always try these first)
                if papers and self.abstract_visualizer:
                    logger.info("Creating abstract-optimized visualizations...")
                    abstract_files = self.abstract_visualizer.create_abstract_dashboard(results, query)
                    visualization_files.extend(abstract_files)

                # Enhanced visualizations (if data is rich enough)
                if papers and self.visualizer:
                    logger.info("Creating enhanced visualizations...")
                    try:
                        enhanced_files = self.visualizer.create_comprehensive_dashboard(results, query)
                        visualization_files.extend(enhanced_files)
                    except Exception as e:
                        logger.warning(f"Enhanced visualizations partially failed: {e}")

                # LLM analysis visualizations
                if results["llm_analysis"] and not results["llm_analysis"].get("error") and self.visualizer:
                    logger.info("Creating LLM analysis visualizations...")
                    llm_files = self.visualizer.create_llm_analysis_dashboard(results["llm_analysis"], query)
                    visualization_files.extend(llm_files)

                results["visualizations"] = visualization_files
                logger.info(f"✅ Generated {len(visualization_files)} visualization files")

            except Exception as e:
                logger.error(f"❌ Visualization generation failed: {e}")
                results["visualizations"] = []

        return results

    def get_analysis_summary(self, papers: List[Paper]) -> dict:
        """Generate summary statistics for the analysis"""
        total_papers = len(papers)
        with_abstracts = sum(1 for p in papers if getattr(p, 'abstract', None))

        # Only calculate PDF-related stats if not in abstract-only mode
        if not self.skip_pdf_download:
            with_pmcids = sum(1 for p in papers if p.pmcid)
            with_fulltext_potential = sum(1 for p in papers if p.has_fulltext)
            downloaded_pdfs = sum(1 for p in papers if p.download_success)
            extracted_text = sum(1 for p in papers if p.full_text)

            return {
                "total_papers": total_papers,
                "with_abstracts": with_abstracts,
                "with_pmcids": with_pmcids,
                "with_fulltext_potential": with_fulltext_potential,
                "downloaded_pdfs": downloaded_pdfs,
                "extracted_text": extracted_text,
                "mode": "full_paper",
                "success_rates": {
                    "abstract_coverage": (with_abstracts / total_papers) * 100 if total_papers > 0 else 0,
                    "pmcid_conversion": (with_pmcids / total_papers) * 100 if total_papers > 0 else 0,
                    "pdf_download": (downloaded_pdfs / with_fulltext_potential) * 100 if with_fulltext_potential > 0 else 0,
                    "text_extraction": (extracted_text / downloaded_pdfs) * 100 if downloaded_pdfs > 0 else 0,
                },
            }
        else:
            return {
                "total_papers": total_papers,
                "with_abstracts": with_abstracts,
                "mode": "abstract_only",
                "success_rates": {
                    "abstract_coverage": (with_abstracts / total_papers) * 100 if total_papers > 0 else 0,
                },
            }


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Modular PubMed Literature Analysis Pipeline"
    )

    parser.add_argument(
        "--query", "-q", type=str, required=True, help="PubMed search query"
    )

    parser.add_argument(
        "--max-papers",
        "-n",
        type=int,
        default=50,
        help="Maximum number of papers to analyze (default: 50)",
    )

    parser.add_argument(
        "--start-date", type=str, help="Start date filter (YYYY/MM/DD format)"
    )

    parser.add_argument(
        "--end-date", type=str, help="End date filter (YYYY/MM/DD format)"
    )

    parser.add_argument(
        "--email", "-e", type=str, help="Email address for NCBI API (overrides default)"
    )

    parser.add_argument(
        "--full-paper",
        action="store_true",
        help="Enable full-paper mode with PDF downloading (default: abstract-only)"
    )

    parser.add_argument(
        "--llm-analysis",
        action="store_true",
        help="Enable LLM-powered analysis (requires OpenAI or DeepSeek API key)"
    )

    parser.add_argument(
        "--visualizations",
        action="store_true",
        help="Generate comprehensive visualizations"
    )

    return parser.parse_args()


async def main():
    """Main execution function"""
    args = parse_arguments()

    # Configuration - CHANGE THE EMAIL ADDRESS TO YOUR OWN
    EMAIL = args.email if args.email else "mrasic2@uic.edu"
    NCBI_API_KEY = os.getenv("NCBI_API_KEY")
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")

    # Validate email
    if EMAIL == "mrasic2@uic.edu" and not args.email:
        logger.warning(
            "⚠️  Using default email address. Please change it in main.py or use --email flag"
        )
        logger.warning("⚠️  NCBI requires a valid email address for API access")

    # Initialize pipeline
    skip_pdf_download = not args.full_paper  # Default to abstract-only unless --full-paper is specified

    if skip_pdf_download:
        logger.info("🔍 Running in ABSTRACT-ONLY mode (faster, no PDF downloads)")
        logger.info("💡 Use --full-paper flag to enable PDF downloading")
    else:
        logger.info("📄 Running in FULL-PAPER mode (includes PDF downloads)")

    pipeline = ModularPubMedPipeline(
        email=EMAIL,
        api_key=NCBI_API_KEY,
        openai_key=OPENAI_KEY,
        deepseek_key=DEEPSEEK_KEY,
        skip_pdf_download=skip_pdf_download,
    )

    try:
        # Run the analysis pipeline
        results = await pipeline.run_pipeline(
            query=args.query,
            max_papers=args.max_papers,
            start_date=args.start_date,
            end_date=args.end_date,
            enable_llm_analysis=args.llm_analysis,
            enable_visualizations=args.visualizations,
        )

        papers = results["papers"]

        if not papers:
            logger.error("No papers found or processed")
            return

        # Generate and display summary
        summary = results["pipeline_summary"]

        logger.info("📊 Analysis Summary:")
        logger.info(f"   Total papers: {summary['total_papers']}")
        logger.info(f"   With abstracts: {summary['with_abstracts']} ({summary['success_rates']['abstract_coverage']:.1f}%)")

        if summary['mode'] == 'full_paper':
            logger.info(
                f"   PMC IDs found: {summary['with_pmcids']} ({summary['success_rates']['pmcid_conversion']:.1f}%)"
            )
            logger.info(
                f"   PDFs downloaded: {summary['downloaded_pdfs']} ({summary['success_rates']['pdf_download']:.1f}%)"
            )
            logger.info(
                f"   Text extracted: {summary['extracted_text']} ({summary['success_rates']['text_extraction']:.1f}%)"
            )
        else:
            logger.info(f"   Mode: Pure abstract analysis (no downloads)")

        # Report LLM analysis results
        if results["llm_analysis"] and not results["llm_analysis"].get("error"):
            llm_summary = results["llm_analysis"].get("batch_analysis_results", {}).get("summary", {})
            total_llm_analyses = llm_summary.get("total_analyses", 0)
            unique_papers_analyzed = llm_summary.get("unique_papers", 0)
            logger.info(f"   LLM analyses: {total_llm_analyses} (across {unique_papers_analyzed} papers)")

        # Report visualization results
        if results["visualizations"]:
            logger.info(f"   Visualizations: {len(results['visualizations'])} files generated")

        logger.info("🎉 Enhanced pipeline execution complete!")

        # Save comprehensive results
        import json

        output_results = {
            "query": results["query"],
            "pipeline_summary": summary,
            "papers": [
                {
                    "pmid": p.pmid,
                    "pmcid": p.pmcid,
                    "title": p.title,
                    "journal": p.journal,
                    "abstract": p.abstract,
                    "has_fulltext": p.has_fulltext,
                    "download_success": p.download_success,
                    "pdf_path": p.pdf_path,
                    "error_message": p.error_message,
                }
                for p in papers
            ],
            "llm_analysis_summary": results["llm_analysis"] if results["llm_analysis"] and not results["llm_analysis"].get("error") else None,
            "visualization_files": results["visualizations"]
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"enhanced_pipeline_results_{timestamp}.json"

        with open(results_filename, "w") as f:
            json.dump(output_results, f, indent=2, default=str)

        logger.info(f"Results saved to {results_filename}")

        # Save detailed LLM results if available
        if results["llm_analysis"] and results["llm_analysis"].get("detailed_results"):
            if hasattr(pipeline, 'llm_analyzer') and pipeline.llm_analyzer:
                llm_filename = pipeline.llm_analyzer.save_results(f"detailed_llm_analysis_{timestamp}.json")
                logger.info(f"Detailed LLM analysis saved to {llm_filename}")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
