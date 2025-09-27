#!/usr/bin/env python3
"""
Simple test for section-aware RAG functionality
"""

import logging
import asyncio
from pubmed_analyzer.core.section_aware_rag import SectionAwareRAGAnalyzer, QueryType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_section_aware_rag():
    """Test basic section-aware RAG functionality"""
    logger.info("🧪 Testing Section-Aware RAG System")

    # Sample paper data
    sample_papers = [
        {
            'pmid': 'TEST001',
            'title': 'Machine Learning in Medical Diagnosis',
            'authors': ['Smith, J.', 'Doe, A.'],
            'abstract': 'This study explores machine learning applications in medical diagnosis. We developed a novel algorithm for image classification.',
            'processing_mode': 'test'
        },
        {
            'pmid': 'TEST002',
            'title': 'COVID-19 Treatment Strategies',
            'authors': ['Johnson, M.', 'Brown, K.'],
            'abstract': 'We analyzed various treatment approaches for COVID-19 patients. Results show significant improvements with early intervention.',
            'processing_mode': 'test'
        }
    ]

    try:
        # Initialize analyzer
        analyzer = SectionAwareRAGAnalyzer(storage_path="./test_section_rag")

        # Process papers
        logger.info("📊 Processing sample papers...")
        results = analyzer.process_papers_with_sections(sample_papers)

        logger.info(f"✅ Processing results:")
        logger.info(f"   Papers processed: {results['processed_papers']}")
        logger.info(f"   Total sections: {results['total_sections']}")

        # Test queries
        test_queries = [
            ("machine learning applications", QueryType.CONCEPTUAL),
            ("treatment approaches", QueryType.METHODOLOGICAL),
            ("key findings", QueryType.EMPIRICAL)
        ]

        for query, query_type in test_queries:
            logger.info(f"🔍 Testing query: '{query}' ({query_type.value})")

            result = analyzer.section_aware_query(
                query=query,
                query_type=query_type,
                limit=3
            )

            logger.info(f"   Results: {len(result.get('contexts', []))}")
            logger.info(f"   Answer: {result.get('response', {}).get('answer', 'No answer')[:100]}...")

        # Get statistics
        stats = analyzer.get_section_statistics()
        logger.info(f"📈 System stats: {stats}")

        logger.info("✅ Section-aware RAG test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_section_aware_rag())
    exit(0 if success else 1)