#!/usr/bin/env python3
"""
Hybrid RAG System Demo
Interactive demonstration of DeepSeek + ChromaDB integration
"""

import asyncio
import os
import json
import logging
from typing import Dict, List, Any
from datetime import datetime

# Import the hybrid system
from pubmed_analyzer.core.hybrid_rag_system import (
    HybridRAGSystem, QueryContext, QueryType, ScoringStrategy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridRAGDemo:
    """Interactive demo interface for the Hybrid RAG System"""

    def __init__(self):
        self.system = None
        self.session_history = []

    async def initialize_system(self, chromadb_path: str = "./demo_chromadb"):
        """Initialize the hybrid RAG system"""
        print("🚀 Initializing Hybrid RAG System...")

        # Check for DeepSeek API key
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_key:
            print("⚠️  Warning: DEEPSEEK_API_KEY not found in environment variables")
            print("   Set it with: export DEEPSEEK_API_KEY='your_api_key_here'")
            print("   The demo will work for retrieval but not LLM generation")

        try:
            self.system = HybridRAGSystem(
                chromadb_path=chromadb_path,
                deepseek_api_key=deepseek_key
            )
            print("✅ Hybrid RAG System initialized successfully!")
            return True

        except Exception as e:
            print(f"❌ Failed to initialize system: {e}")
            return False

    async def demo_with_sample_data(self):
        """Demo with sample scientific papers"""
        print("\n📚 Loading sample scientific papers...")

        # Sample papers for demonstration
        sample_papers = [
            {
                'pmid': 'DEMO_001',
                'title': 'Deep Learning Applications in Medical Image Analysis: A Comprehensive Survey',
                'authors': ['Zhang, Wei', 'Smith, John', 'Chen, Li'],
                'journal': 'Nature Biomedical Engineering',
                'year': 2023,
                'abstract': """
                Background: Deep learning has revolutionized medical image analysis, enabling automated diagnosis and treatment planning.
                Methods: We systematically reviewed 200 studies published between 2020-2023, analyzing CNN, transformer, and hybrid architectures.
                Results: Deep learning models achieved 94.2% average accuracy across imaging modalities, with transformers showing superior performance in multi-modal tasks.
                Conclusions: Deep learning demonstrates significant clinical potential, though challenges remain in interpretability and regulatory approval.
                """,
                'sections': {
                    'Introduction': """
                    Medical imaging plays a crucial role in modern healthcare, providing clinicians with essential diagnostic information.
                    Traditional image analysis relies on manual interpretation, which is time-consuming and subject to inter-observer variability.
                    Deep learning algorithms, particularly convolutional neural networks (CNNs), have shown remarkable success in automating medical image analysis tasks.
                    """,
                    'Methods': """
                    We conducted a systematic literature review following PRISMA guidelines. Our search included PubMed, IEEE Xplore, and arXiv databases.
                    Search terms combined "deep learning", "medical imaging", "CNN", "transformer", and specific imaging modalities.
                    We analyzed model architectures, datasets, performance metrics, and clinical validation studies.
                    Data extraction included accuracy, sensitivity, specificity, and deployment status in clinical settings.
                    """,
                    'Results': """
                    A total of 200 studies met inclusion criteria. CNN architectures dominated (65%), followed by transformer models (23%) and hybrid approaches (12%).
                    Average diagnostic accuracy was 94.2% across all studies, with chest X-ray analysis showing highest performance (96.8%).
                    Transformer models achieved superior results in multi-modal imaging tasks, reaching 97.1% accuracy vs 92.3% for traditional CNNs.
                    Clinical deployment was reported in only 15% of studies, highlighting the translation gap.
                    """,
                    'Discussion': """
                    Our findings demonstrate the maturation of deep learning in medical imaging, with consistently high performance across diverse tasks.
                    However, several challenges persist: model interpretability remains limited, regulatory pathways are unclear, and clinical validation is often insufficient.
                    Future research should prioritize explainable AI methods, prospective clinical trials, and standardized evaluation frameworks.
                    The integration of multimodal data and real-world deployment strategies will be crucial for clinical translation.
                    """
                }
            },
            {
                'pmid': 'DEMO_002',
                'title': 'CRISPR-Cas9 Gene Editing: Safety Profile and Clinical Applications',
                'authors': ['Garcia, Maria', 'Johnson, Robert', 'Patel, Anish'],
                'journal': 'Cell',
                'year': 2023,
                'abstract': """
                Background: CRISPR-Cas9 gene editing has emerged as a promising therapeutic approach for genetic disorders.
                Methods: Meta-analysis of 50 clinical trials involving CRISPR gene editing from 2018-2023.
                Results: Off-target editing occurred in 2.1% of cases. Serious adverse events were rare (0.6%).
                Conclusions: CRISPR demonstrates acceptable safety profile for clinical applications with proper safeguards.
                """,
                'sections': {
                    'Background': """
                    CRISPR-Cas9 represents a revolutionary gene editing technology that enables precise modification of DNA sequences.
                    Clinical applications range from treating inherited genetic disorders to developing new cancer therapies.
                    Safety concerns regarding off-target effects and unintended consequences have been primary barriers to widespread adoption.
                    """,
                    'Methods': """
                    We conducted a comprehensive meta-analysis of CRISPR clinical trials registered in ClinicalTrials.gov and international databases.
                    Inclusion criteria required completed or ongoing Phase I/II trials with published safety data.
                    Primary endpoints included off-target editing rates, adverse events, and therapeutic efficacy measures.
                    Data extraction followed PRISMA guidelines with independent review by two investigators.
                    """,
                    'Results': """
                    Fifty clinical trials met inclusion criteria, encompassing 1,247 patients across diverse indications.
                    Off-target editing was detected in 2.1% of cases using sensitive detection methods.
                    Serious adverse events directly attributable to CRISPR editing occurred in 0.6% of patients.
                    Therapeutic efficacy was demonstrated in 78% of trials, with highest success rates in monogenic disorders.
                    """
                }
            }
        ]

        # Process sample papers
        try:
            print("🔄 Processing sample papers into ChromaDB...")
            results = self.system.section_rag.process_papers_with_sections(sample_papers)
            print(f"✅ Processed {results['processed_papers']} papers with {results['total_sections']} sections")
            return True

        except Exception as e:
            print(f"❌ Failed to process sample papers: {e}")
            return False

    async def interactive_demo(self):
        """Run interactive demo"""
        print("\n🤖 Hybrid RAG System Demo")
        print("=" * 50)
        print("Ask research questions and see the hybrid retrieval + LLM in action!")
        print("Type 'quit' to exit, 'help' for commands, 'stats' for system statistics")
        print()

        while True:
            try:
                # Get user input
                question = input("🔬 Research Question: ").strip()

                if question.lower() == 'quit':
                    print("👋 Goodbye!")
                    break

                elif question.lower() == 'help':
                    self.show_help()
                    continue

                elif question.lower() == 'stats':
                    await self.show_stats()
                    continue

                elif question.lower().startswith('demo'):
                    await self.run_demo_queries()
                    continue

                elif not question:
                    continue

                # Process the question
                await self.process_question(question)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    async def process_question(self, question: str):
        """Process a research question"""
        print(f"\n🔍 Processing: {question}")
        print("-" * 50)

        start_time = datetime.now()

        try:
            # Determine query type (simple heuristic)
            query_type = self.determine_query_type(question)
            print(f"📊 Query Type: {query_type.value}")

            # Answer the question
            if self.system.deepseek_client:
                response = await self.system.answer_question(
                    question=question,
                    query_type=query_type,
                    scoring_strategy=ScoringStrategy.ADAPTIVE
                )

                # Display results
                self.display_response(response)

            else:
                # Retrieval only (no LLM)
                print("⚠️  LLM not available - showing retrieval results only")

                query_context = QueryContext(
                    original_query=question,
                    processed_query=question,
                    query_type=query_type,
                    scoring_strategy=ScoringStrategy.ADAPTIVE
                )

                results = await self.system.hybrid_retrieve(query_context)
                self.display_retrieval_results(results)

            # Save to history
            processing_time = (datetime.now() - start_time).total_seconds()
            self.session_history.append({
                "question": question,
                "timestamp": start_time.isoformat(),
                "processing_time": processing_time
            })

        except Exception as e:
            print(f"❌ Error processing question: {e}")

    def determine_query_type(self, question: str) -> QueryType:
        """Simple heuristic to determine query type"""
        question_lower = question.lower()

        if any(word in question_lower for word in ["method", "approach", "technique", "protocol", "procedure"]):
            return QueryType.METHODOLOGICAL
        elif any(word in question_lower for word in ["result", "finding", "outcome", "effect", "performance"]):
            return QueryType.EMPIRICAL
        elif any(word in question_lower for word in ["concept", "theory", "principle", "background", "overview"]):
            return QueryType.CONCEPTUAL
        elif any(word in question_lower for word in ["compare", "contrast", "difference", "versus", "vs"]):
            return QueryType.COMPARATIVE
        else:
            return QueryType.SYNTHESIS

    def display_response(self, response):
        """Display LLM response with formatting"""
        print(f"🤖 **Answer** (Confidence: {response.confidence:.1%}):")
        print(response.answer)
        print()

        if response.sources:
            print(f"📚 **Sources** ({len(response.sources)} papers):")
            for i, source in enumerate(response.sources[:3], 1):  # Show top 3
                print(f"   {i}. {source.title}")
                print(f"      {source.journal} ({source.year}) - Score: {source.final_score:.2f}")
                print(f"      Section: {source.section_type}")
                print()

        if response.limitations:
            print("⚠️  **Limitations:**")
            for limitation in response.limitations[:3]:
                print(f"   • {limitation}")
            print()

        if response.follow_up_questions:
            print("🔍 **Follow-up Questions:**")
            for question in response.follow_up_questions[:3]:
                print(f"   • {question}")
            print()

        print(f"⏱️  Generation Time: {response.generation_time:.2f}s")

    def display_retrieval_results(self, results):
        """Display retrieval results only"""
        print(f"📚 **Retrieved Results** ({len(results)} documents):")

        for i, result in enumerate(results[:5], 1):
            print(f"\n{i}. **{result.title}**")
            print(f"   Authors: {', '.join(result.authors[:3])}{'...' if len(result.authors) > 3 else ''}")
            print(f"   Journal: {result.journal} ({result.year})")
            print(f"   Section: {result.section_type}")
            print(f"   Score: {result.final_score:.2f} (V:{result.vector_score:.2f}, K:{result.keyword_score:.2f}, S:{result.section_relevance_score:.2f})")
            print(f"   Content: {result.content[:200]}...")

    async def show_stats(self):
        """Show system statistics"""
        print("\n📊 System Statistics:")
        print("-" * 30)

        stats = self.system.get_system_stats()

        print(f"Embedding Cache: {stats['embedding_cache_size']} entries")
        print(f"DeepSeek Connected: {'✅' if stats['deepseek_connected'] else '❌'}")

        if stats['section_rag_stats']:
            rag_stats = stats['section_rag_stats']
            print(f"ChromaDB Collections: {rag_stats.get('collections', {})}")

        print(f"Session Questions: {len(self.session_history)}")
        print()

    def show_help(self):
        """Show help information"""
        print("\n🤖 Hybrid RAG System Commands:")
        print("-" * 40)
        print("help          - Show this help message")
        print("stats         - Show system statistics")
        print("demo          - Run demonstration queries")
        print("quit          - Exit the demo")
        print()
        print("Research Question Examples:")
        print("• What machine learning methods are used in medical imaging?")
        print("• How effective is CRISPR gene editing for genetic disorders?")
        print("• What are the main limitations of deep learning in healthcare?")
        print("• Compare CNN vs transformer architectures in medical AI")
        print()

    async def run_demo_queries(self):
        """Run predefined demo queries"""
        demo_questions = [
            "What machine learning methods are most effective for medical image analysis?",
            "What are the safety concerns with CRISPR gene editing?",
            "How do CNN and transformer models compare in medical imaging tasks?",
            "What are the main limitations of AI in clinical deployment?"
        ]

        print("\n🎬 Running Demo Questions...")
        print("=" * 50)

        for i, question in enumerate(demo_questions, 1):
            print(f"\n📝 Demo Question {i}/{len(demo_questions)}:")
            await self.process_question(question)

            if i < len(demo_questions):
                input("\n[Press Enter to continue...]")


async def main():
    """Main demo function"""
    demo = HybridRAGDemo()

    print("🧬 PubMed Analyzer - Hybrid RAG System Demo")
    print("=" * 60)

    # Initialize system
    if not await demo.initialize_system():
        return

    # Load sample data
    if not await demo.demo_with_sample_data():
        return

    # Run interactive demo
    await demo.interactive_demo()


if __name__ == "__main__":
    asyncio.run(main())