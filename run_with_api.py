#!/usr/bin/env python3
"""
Run hybrid RAG with DeepSeek API key
"""

import os
from quick_hybrid_demo import SimpleHybridRAG

def test_with_llm():
    """Test with LLM if API key is available"""
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        print("❌ Please set DEEPSEEK_API_KEY environment variable")
        print("   export DEEPSEEK_API_KEY='your_key_here'")
        return

    print("🤖 Testing with DeepSeek LLM...")
    system = SimpleHybridRAG(api_key)

    # Test question
    question = "What are the main advantages of deep learning in medical imaging according to recent research?"

    result = system.answer_question(question)

    print(f"\n🔬 Question: {question}")
    print(f"🤖 Answer: {result['answer']}")
    print(f"📚 Sources: {len(result['sources'])} papers")
    if result.get('generation_time'):
        print(f"⏱️ Generation time: {result['generation_time']:.2f}s")

if __name__ == "__main__":
    test_with_llm()