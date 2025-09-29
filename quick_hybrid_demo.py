#!/usr/bin/env python3
"""
Quick Hybrid RAG Demo
Simplified version that works with existing dependencies
"""

import os
import json
import time
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Use built-in urllib instead of requests for now
import urllib.request
import urllib.parse
import urllib.error


@dataclass
class SimpleResult:
    """Simple result structure"""
    content: str
    title: str
    score: float
    source: str


class SimpleDeepSeekClient:
    """Simplified DeepSeek client using urllib"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"

    def generate_response(self, prompt: str, max_tokens: int = 1500) -> Dict[str, Any]:
        """Generate response from DeepSeek API"""
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }

        try:
            # Prepare request
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self.base_url,
                data=data,
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
            )

            # Make request
            start_time = time.time()
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode())
                generation_time = time.time() - start_time

                return {
                    "success": True,
                    "content": result["choices"][0]["message"]["content"],
                    "generation_time": generation_time
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "generation_time": 0.0
            }


class SimpleHybridRAG:
    """Simplified hybrid RAG system"""

    def __init__(self, deepseek_api_key: Optional[str] = None):
        self.deepseek_key = deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
        self.deepseek_client = SimpleDeepSeekClient(self.deepseek_key) if self.deepseek_key else None

        # Sample scientific papers database
        self.papers_db = [
            {
                "title": "Deep Learning Applications in Medical Image Analysis: A Comprehensive Survey",
                "authors": ["Zhang, Wei", "Smith, John", "Chen, Li"],
                "journal": "Nature Biomedical Engineering",
                "year": 2023,
                "content": """
                Background: Deep learning has revolutionized medical image analysis, enabling automated diagnosis and treatment planning.
                Methods: We systematically reviewed 200 studies published between 2020-2023, analyzing CNN, transformer, and hybrid architectures.
                The most effective approaches combined convolutional neural networks with attention mechanisms.
                ResNet, DenseNet, and Vision Transformers showed superior performance across multiple imaging modalities.
                Results: Deep learning models achieved 94.2% average accuracy across imaging modalities, with transformers showing
                superior performance in multi-modal tasks (96.1% vs 91.8% for traditional CNNs).
                Conclusions: Deep learning demonstrates significant clinical potential, though challenges remain in interpretability
                and regulatory approval. Future work should focus on explainable AI and prospective clinical trials.
                """,
                "section_type": "comprehensive"
            },
            {
                "title": "CRISPR-Cas9 Gene Editing: Safety Profile and Clinical Applications",
                "authors": ["Garcia, Maria", "Johnson, Robert", "Patel, Anish"],
                "journal": "Cell",
                "year": 2023,
                "content": """
                Background: CRISPR-Cas9 gene editing has emerged as a promising therapeutic approach for genetic disorders.
                Methods: Meta-analysis of 50 clinical trials involving CRISPR gene editing from 2018-2023.
                We analyzed off-target effects, adverse events, and therapeutic efficacy across diverse applications.
                Results: Off-target editing occurred in 2.1% of cases using sensitive detection methods.
                Serious adverse events directly attributable to CRISPR editing occurred in 0.6% of patients.
                Therapeutic efficacy was demonstrated in 78% of trials, with highest success rates in monogenic disorders.
                Discussion: CRISPR demonstrates acceptable safety profile for clinical applications with proper safeguards.
                Future applications should focus on improved delivery mechanisms and enhanced specificity.
                """,
                "section_type": "clinical"
            },
            {
                "title": "Machine Learning in Drug Discovery: Current Applications and Future Prospects",
                "authors": ["Brown, Alice", "Davis, Michael", "Wilson, Sarah"],
                "journal": "Nature Reviews Drug Discovery",
                "year": 2023,
                "content": """
                Introduction: Machine learning (ML) has transformed drug discovery by accelerating target identification,
                lead optimization, and clinical trial design.
                Methods: We analyzed 150 drug discovery programs utilizing ML between 2019-2023.
                Random forests, neural networks, and graph neural networks were the most common approaches.
                Results: ML-assisted drug discovery programs showed 35% faster progression to clinical trials.
                Success rates improved from 12% to 18% for programs incorporating ML early in development.
                Graph neural networks achieved highest accuracy (87%) for molecular property prediction.
                Limitations: Data quality, model interpretability, and regulatory acceptance remain challenges.
                Conclusions: ML integration is becoming essential for competitive drug discovery.
                """,
                "section_type": "drug_discovery"
            }
        ]

        print("🤖 Simple Hybrid RAG System initialized")
        print(f"   Papers Database: {len(self.papers_db)} papers")
        print(f"   DeepSeek: {'✅ Connected' if self.deepseek_client else '❌ No API key'}")

    def simple_retrieve(self, query: str, max_results: int = 3) -> List[SimpleResult]:
        """Simple keyword-based retrieval"""
        query_terms = query.lower().split()
        results = []

        for paper in self.papers_db:
            # Simple scoring based on keyword matches
            content = paper["content"].lower()
            title = paper["title"].lower()

            # Count keyword matches
            content_matches = sum(1 for term in query_terms if term in content)
            title_matches = sum(1 for term in query_terms if term in title) * 2  # Title boost

            total_matches = content_matches + title_matches

            if total_matches > 0:
                # Simple scoring
                score = total_matches / len(query_terms)

                results.append(SimpleResult(
                    content=paper["content"][:500] + "...",
                    title=paper["title"],
                    score=score,
                    source=f"{paper['journal']} ({paper['year']})"
                ))

        # Sort by score and return top results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer question using simple retrieval + DeepSeek"""
        print(f"\n🔍 Processing: {question}")
        print("-" * 50)

        # Retrieve relevant papers
        results = self.simple_retrieve(question)
        print(f"📚 Found {len(results)} relevant papers")

        if not results:
            return {
                "answer": "No relevant papers found in the database.",
                "sources": [],
                "error": None
            }

        if not self.deepseek_client:
            return {
                "answer": "DeepSeek API key not configured. Here are the retrieved papers:",
                "sources": results,
                "error": "No API key"
            }

        # Build context
        context = self._build_context(results)

        # Generate prompt
        prompt = f"""You are a scientific research assistant. Answer the following question based on the provided research papers.

Question: {question}

Scientific Papers:
{context}

Instructions:
1. Provide a comprehensive answer based on the papers
2. Cite specific papers when making claims
3. Mention any limitations or conflicting information
4. Keep the answer focused and evidence-based

Answer:"""

        # Generate response
        llm_result = self.deepseek_client.generate_response(prompt)

        if llm_result["success"]:
            return {
                "answer": llm_result["content"],
                "sources": results,
                "generation_time": llm_result["generation_time"],
                "error": None
            }
        else:
            return {
                "answer": f"Error generating response: {llm_result['error']}",
                "sources": results,
                "error": llm_result["error"]
            }

    def _build_context(self, results: List[SimpleResult]) -> str:
        """Build context from retrieval results"""
        context_parts = []

        for i, result in enumerate(results, 1):
            context_parts.append(f"""
Paper {i}: {result.title}
Source: {result.source}
Relevance Score: {result.score:.2f}

Content: {result.content}

---""")

        return "\n".join(context_parts)

    def interactive_demo(self):
        """Run interactive demo"""
        print("\n🔬 Simple Hybrid RAG Demo")
        print("=" * 40)
        print("Ask research questions about:")
        print("• Machine learning in medical imaging")
        print("• CRISPR gene editing safety")
        print("• ML in drug discovery")
        print()
        print("Type 'quit' to exit")
        print()

        while True:
            try:
                question = input("❓ Question: ").strip()

                if question.lower() == 'quit':
                    print("👋 Goodbye!")
                    break

                if not question:
                    continue

                # Process question
                result = self.answer_question(question)

                # Display answer
                print("\n🤖 Answer:")
                print(result["answer"])

                if result["sources"]:
                    print(f"\n📚 Sources ({len(result['sources'])}):")
                    for i, source in enumerate(result["sources"], 1):
                        print(f"   {i}. {source.title}")
                        print(f"      {source.source} - Score: {source.score:.2f}")

                if result.get("generation_time"):
                    print(f"\n⏱️  Response time: {result['generation_time']:.2f}s")

                print("\n" + "="*50)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def demo_questions(self):
        """Run demo with preset questions"""
        questions = [
            "What machine learning methods are most effective for medical image analysis?",
            "How safe is CRISPR gene editing for clinical applications?",
            "What role does machine learning play in drug discovery?",
            "What are the main limitations of AI in healthcare?"
        ]

        print("\n🎬 Demo Questions")
        print("=" * 40)

        for i, question in enumerate(questions, 1):
            print(f"\n📝 Question {i}: {question}")
            result = self.answer_question(question)

            print(f"🤖 Answer: {result['answer'][:200]}...")
            print(f"📚 Sources: {len(result['sources'])} papers")

            if i < len(questions):
                input("\n[Press Enter for next question...]")


def main():
    """Main function"""
    print("🧬 Simple Hybrid RAG System Demo")
    print("=" * 50)

    # Check for API key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠️  Warning: DEEPSEEK_API_KEY not found")
        print("   Set it with: export DEEPSEEK_API_KEY='your_key'")
        print("   The demo will work for retrieval but not LLM generation")
        print()

    # Initialize system
    system = SimpleHybridRAG(api_key)

    # Ask user what they want to do
    print("\nChoose an option:")
    print("1. Interactive demo")
    print("2. Run preset demo questions")
    print("3. Quick test")

    try:
        choice = input("\nEnter choice (1/2/3): ").strip()

        if choice == "1":
            system.interactive_demo()
        elif choice == "2":
            system.demo_questions()
        elif choice == "3":
            # Quick test
            result = system.answer_question("What are the benefits of deep learning in medical imaging?")
            print("\n🤖 Quick Test Result:")
            print(result["answer"])
        else:
            print("Invalid choice. Running interactive demo...")
            system.interactive_demo()

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()