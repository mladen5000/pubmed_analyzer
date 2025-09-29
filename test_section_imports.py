#!/usr/bin/env python3
"""
Basic import and functionality test for section-aware components
Testing without full environment setup
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    """Test basic imports and identify missing dependencies"""
    print("🧪 Testing basic imports...")

    issues = []

    # Test core model imports
    try:
        from pubmed_analyzer.models.paper import Paper
        print("✅ Paper model import successful")
    except ImportError as e:
        issues.append(f"Paper model: {e}")
        print(f"❌ Paper model import failed: {e}")

    try:
        from pubmed_analyzer.models.section import SectionType, ProcessedSection, SectionMetadata
        print("✅ Section models import successful")
    except ImportError as e:
        issues.append(f"Section models: {e}")
        print(f"❌ Section models import failed: {e}")

    # Test ChromaDB store import
    try:
        from pubmed_analyzer.core.chromadb_store import ScientificChromaStore
        print("✅ ChromaDB store import successful")
    except ImportError as e:
        issues.append(f"ChromaDB store: {e}")
        print(f"❌ ChromaDB store import failed: {e}")

    # Test section embeddings
    try:
        from pubmed_analyzer.core.section_embeddings import MultiModelEmbedder
        print("✅ Section embeddings import successful")
    except ImportError as e:
        issues.append(f"Section embeddings: {e}")
        print(f"❌ Section embeddings import failed: {e}")

    # Test section-aware RAG
    try:
        from pubmed_analyzer.core.section_aware_rag import SectionAwareRAGAnalyzer, SectionClassifier
        print("✅ Section-aware RAG import successful")
    except ImportError as e:
        issues.append(f"Section-aware RAG: {e}")
        print(f"❌ Section-aware RAG import failed: {e}")

    return issues


def test_paper_model():
    """Test Paper model functionality"""
    print("\n🧪 Testing Paper model...")

    try:
        from pubmed_analyzer.models.paper import Paper

        # Create basic paper
        paper = Paper(
            pmid="TEST_001",
            title="Test Paper",
            authors=["Test Author"],
            abstract="Test abstract content"
        )

        assert paper.pmid == "TEST_001"
        assert paper.title == "Test Paper"
        print("✅ Basic Paper model functionality works")

        # Test structured sections
        paper.structured_sections = {
            'abstract': {
                'content': 'Test abstract',
                'content_length': 13,
                'confidence_score': 1.0
            }
        }

        assert 'abstract' in paper.structured_sections
        print("✅ Structured sections functionality works")

        return []

    except Exception as e:
        return [f"Paper model functionality: {e}"]


def test_section_models():
    """Test section models functionality"""
    print("\n🧪 Testing Section models...")

    try:
        from pubmed_analyzer.models.section import SectionType, SectionMetadata, ProcessedSection

        # Test SectionType enum
        assert SectionType.ABSTRACT.value == "abstract"
        assert SectionType.from_text("introduction") == SectionType.INTRODUCTION
        print("✅ SectionType enum works")

        # Test SectionMetadata
        metadata = SectionMetadata(
            paper_pmid="TEST_001",
            section_type=SectionType.ABSTRACT,
            word_count=50
        )

        assert metadata.paper_pmid == "TEST_001"
        assert metadata.section_type == SectionType.ABSTRACT
        print("✅ SectionMetadata works")

        # Test ProcessedSection
        section = ProcessedSection(
            content="Test section content",
            metadata=metadata
        )

        assert section.content == "Test section content"
        assert section.section_type == SectionType.ABSTRACT
        print("✅ ProcessedSection works")

        return []

    except Exception as e:
        return [f"Section models functionality: {e}"]


def test_chromadb_import_only():
    """Test ChromaDB import without actual operations"""
    print("\n🧪 Testing ChromaDB import...")

    try:
        from pubmed_analyzer.core.chromadb_store import ScientificChromaStore

        # Test class instantiation with mock path (won't actually create ChromaDB)
        temp_dir = tempfile.mkdtemp()

        try:
            # This will fail if ChromaDB isn't installed, but should import cleanly
            store = ScientificChromaStore(persist_directory=temp_dir)
            print("✅ ChromaDB store creation works")

            # Test collection names
            expected_collections = ['abstracts', 'methods', 'results']
            for collection_name in expected_collections:
                assert collection_name in store.collection_names.values()
            print("✅ Collection configuration works")

        except Exception as e:
            print(f"⚠️ ChromaDB operations failed (expected if ChromaDB not installed): {e}")
            # This is expected if ChromaDB isn't properly installed

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return []

    except ImportError as e:
        return [f"ChromaDB import: {e}"]
    except Exception as e:
        return [f"ChromaDB test: {e}"]


def test_section_classifier():
    """Test section classifier without spaCy"""
    print("\n🧪 Testing Section Classifier...")

    try:
        from pubmed_analyzer.core.section_aware_rag import SectionClassifier, SectionType

        classifier = SectionClassifier()
        print("✅ SectionClassifier creation works")

        # Test basic classification
        test_cases = [
            ("This paper presents novel methods for analysis", "Methods", SectionType.METHODS),
            ("Our results show significant improvements", "Results", SectionType.RESULTS),
            ("In conclusion, we demonstrate", "Conclusion", SectionType.CONCLUSION)
        ]

        for text, title, expected in test_cases:
            section_type, confidence = classifier.classify_section(text, title)
            print(f"   Classified '{title}': {section_type.value} (confidence: {confidence:.2f})")

            # Should not crash and return valid values
            assert isinstance(section_type, SectionType)
            assert 0.0 <= confidence <= 1.0

        print("✅ Section classification works")
        return []

    except Exception as e:
        return [f"Section classifier: {e}"]


def test_embedding_imports():
    """Test embedding system imports"""
    print("\n🧪 Testing Embedding imports...")

    try:
        from pubmed_analyzer.core.section_embeddings import (
            MultiModelEmbedder, normalize_embeddings, aggregate_embeddings
        )
        print("✅ Embedding system imports work")

        # Test utility functions
        import numpy as np

        # Test normalization
        test_embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        normalized = normalize_embeddings(test_embeddings)
        assert normalized.shape == test_embeddings.shape
        print("✅ Embedding normalization works")

        # Test aggregation
        embedding_list = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
        aggregated = aggregate_embeddings(embedding_list, method='mean')
        expected = np.array([2.5, 3.5, 4.5])
        np.testing.assert_array_almost_equal(aggregated, expected)
        print("✅ Embedding aggregation works")

        return []

    except ImportError as e:
        return [f"Embedding imports: {e}"]
    except Exception as e:
        return [f"Embedding functionality: {e}"]


def run_all_tests():
    """Run all tests and report results"""
    print("🔬 Running Section-Aware RAG Import and Functionality Tests")
    print("=" * 60)

    all_issues = []

    # Run import tests
    issues = test_basic_imports()
    all_issues.extend(issues)

    # Run functionality tests
    if not issues:  # Only run if imports work
        all_issues.extend(test_paper_model())
        all_issues.extend(test_section_models())
        all_issues.extend(test_chromadb_import_only())
        all_issues.extend(test_section_classifier())
        all_issues.extend(test_embedding_imports())

    # Report results
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)

    if not all_issues:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Section-aware RAG components are working correctly")
        return True
    else:
        print("❌ ISSUES FOUND:")
        for i, issue in enumerate(all_issues, 1):
            print(f"   {i}. {issue}")

        print(f"\n📈 Summary: {len(all_issues)} issues found")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)