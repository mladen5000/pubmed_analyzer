#!/usr/bin/env python3
"""
Direct import test to bypass dependency chain issues
"""

import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_direct_paper_model():
    """Test Paper model directly"""
    print("🧪 Testing Paper model directly...")

    try:
        # Direct import without going through __init__.py
        sys.path.insert(0, str(Path(__file__).parent / "pubmed_analyzer"))

        from models.paper import Paper

        # Test basic functionality
        paper = Paper(
            pmid="TEST_001",
            title="Test Paper",
            authors=["Test Author"],
            abstract="Test abstract content"
        )

        assert paper.pmid == "TEST_001"
        assert paper.title == "Test Paper"

        # Test structured sections
        paper.structured_sections = {
            'abstract': {
                'content': 'Test abstract',
                'content_length': 13,
                'confidence_score': 1.0
            }
        }

        assert 'abstract' in paper.structured_sections
        print("✅ Paper model works correctly")
        return True

    except Exception as e:
        print(f"❌ Paper model failed: {e}")
        return False

def test_section_models():
    """Test section models directly"""
    print("🧪 Testing Section models directly...")

    try:
        from models.section import SectionType, SectionMetadata, ProcessedSection

        # Test SectionType enum
        assert SectionType.ABSTRACT.value == "abstract"
        assert SectionType.from_text("introduction") == SectionType.INTRODUCTION

        # Test SectionMetadata
        metadata = SectionMetadata(
            paper_pmid="TEST_001",
            section_type=SectionType.ABSTRACT,
            word_count=50
        )

        assert metadata.paper_pmid == "TEST_001"
        assert metadata.section_type == SectionType.ABSTRACT

        # Test ProcessedSection
        section = ProcessedSection(
            content="Test section content",
            metadata=metadata
        )

        assert section.content == "Test section content"
        assert section.section_type == SectionType.ABSTRACT

        print("✅ Section models work correctly")
        return True

    except Exception as e:
        print(f"❌ Section models failed: {e}")
        return False

def test_section_classifier():
    """Test section classifier directly"""
    print("🧪 Testing Section classifier directly...")

    try:
        # Mock spacy to avoid dependency issues
        import unittest.mock

        with unittest.mock.patch('spacy.load') as mock_spacy:
            mock_spacy.side_effect = OSError("Model not found")

            from core.section_aware_rag import SectionClassifier, SectionType

            classifier = SectionClassifier()

            # Test basic classification (should work without spacy)
            section_type, confidence = classifier.classify_section(
                "This paper presents novel methods for analysis",
                "Methods"
            )

            assert isinstance(section_type, SectionType)
            assert 0.0 <= confidence <= 1.0

            print("✅ Section classifier works correctly")
            return True

    except Exception as e:
        print(f"❌ Section classifier failed: {e}")
        return False

def test_chromadb_store():
    """Test ChromaDB store structure"""
    print("🧪 Testing ChromaDB store structure...")

    try:
        import unittest.mock

        # Mock chromadb to avoid installation requirement
        with unittest.mock.patch('chromadb.PersistentClient') as mock_client:
            mock_collection = unittest.mock.Mock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection

            from core.chromadb_store import ScientificChromaStore

            store = ScientificChromaStore(persist_directory="/tmp/test")

            # Test that collections are configured
            expected_collections = ['abstracts', 'methods', 'results']
            for collection_name in expected_collections:
                assert collection_name in store.collection_names.values()

            print("✅ ChromaDB store structure works correctly")
            return True

    except Exception as e:
        print(f"❌ ChromaDB store failed: {e}")
        return False

def test_embedding_utilities():
    """Test embedding utilities"""
    print("🧪 Testing embedding utilities...")

    try:
        import numpy as np
        import unittest.mock

        # Mock torch and transformers
        with unittest.mock.patch.dict('sys.modules', {
            'torch': unittest.mock.Mock(),
            'transformers': unittest.mock.Mock(),
            'sentence_transformers': unittest.mock.Mock()
        }):
            from core.section_embeddings import normalize_embeddings, aggregate_embeddings

            # Test normalization
            test_embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            normalized = normalize_embeddings(test_embeddings)
            assert normalized.shape == test_embeddings.shape

            # Test aggregation
            embedding_list = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
            aggregated = aggregate_embeddings(embedding_list, method='mean')
            expected = np.array([2.5, 3.5, 4.5])
            np.testing.assert_array_almost_equal(aggregated, expected)

            print("✅ Embedding utilities work correctly")
            return True

    except Exception as e:
        print(f"❌ Embedding utilities failed: {e}")
        return False

def run_direct_tests():
    """Run all direct tests"""
    print("🔬 Running Direct Component Tests")
    print("=" * 50)

    results = []

    results.append(("Paper Model", test_direct_paper_model()))
    results.append(("Section Models", test_section_models()))
    results.append(("Section Classifier", test_section_classifier()))
    results.append(("ChromaDB Store", test_chromadb_store()))
    results.append(("Embedding Utilities", test_embedding_utilities()))

    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20s} {status}")

    print(f"\n📈 Summary: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL DIRECT TESTS PASSED!")
        print("✅ Core section-aware components are working")
        return True
    else:
        print(f"⚠️ {total - passed} test(s) failed")
        return False

if __name__ == "__main__":
    success = run_direct_tests()
    sys.exit(0 if success else 1)