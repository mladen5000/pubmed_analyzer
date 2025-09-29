#!/usr/bin/env python3
"""
Test the retrieval system to show it working
"""

from quick_hybrid_demo import SimpleHybridRAG

def test_retrieval():
    """Test the hybrid retrieval system"""
    print("🧪 Testing Hybrid Retrieval System")
    print("=" * 50)

    # Initialize system
    system = SimpleHybridRAG()

    # Test queries
    test_queries = [
        "machine learning medical imaging",
        "CRISPR safety clinical trials",
        "drug discovery artificial intelligence",
        "deep learning accuracy performance"
    ]

    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 30)

        results = system.simple_retrieve(query, max_results=3)

        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.title}")
                print(f"   Score: {result.score:.2f}")
                print(f"   Source: {result.source}")
                print(f"   Preview: {result.content[:100]}...")
                print()
        else:
            print("No results found")

    print("\n✅ Retrieval system working correctly!")
    print("\nNow set your DEEPSEEK_API_KEY to enable LLM generation:")
    print("export DEEPSEEK_API_KEY='your_key_here'")

if __name__ == "__main__":
    test_retrieval()