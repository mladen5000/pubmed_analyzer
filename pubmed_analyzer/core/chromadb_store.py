#!/usr/bin/env python3
"""
ChromaDB Vector Store for Section-Aware RAG
Optimized for scientific literature with section-based retrieval
"""

import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json

import chromadb
from chromadb.config import Settings
import numpy as np

from ..models.paper import Paper

logger = logging.getLogger(__name__)


class ScientificChromaStore:
    """
    ChromaDB-based vector store optimized for scientific literature
    with section-aware capabilities and rich metadata filtering
    """

    def __init__(self, persist_directory: str = "./chromadb_data"):
        """
        Initialize ChromaDB with persistent storage for scientific papers

        Args:
            persist_directory: Directory for persistent storage
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)

        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Collection names for different content types
        self.collection_names = {
            'abstracts': 'scientific_abstracts',
            'introductions': 'scientific_introductions',
            'methods': 'scientific_methods',
            'results': 'scientific_results',
            'discussions': 'scientific_discussions',
            'conclusions': 'scientific_conclusions',
            'citations': 'citation_contexts',
            'figures_tables': 'figure_table_refs'
        }

        # Initialize collections
        self.collections = {}
        self._setup_collections()

        logger.info(f"🗄️ ChromaDB initialized with persistent storage at {persist_directory}")

    def _setup_collections(self):
        """Setup section-specific collections with optimized metadata"""

        # Common metadata fields for all scientific content
        base_metadata = {
            "pmid": "str",
            "pmcid": "str",
            "title": "str",
            "authors": "list",
            "journal": "str",
            "year": "int",
            "doi": "str"
        }

        # Section-specific metadata extensions
        section_metadata = {
            "section_type": "str",
            "section_order": "int",
            "content_length": "int",
            "confidence_score": "float",
            "has_citations": "bool",
            "has_figures": "bool",
            "research_methodology": "str",
            "finding_type": "str",
            "limitation_mentioned": "bool"
        }

        for section_type, collection_name in self.collection_names.items():
            try:
                # Get or create collection with section-optimized metadata
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"section_type": section_type, "description": f"Scientific {section_type} with rich metadata"}
                )
                self.collections[section_type] = collection
                logger.info(f"✅ Collection '{collection_name}' ready")

            except Exception as e:
                logger.error(f"❌ Failed to setup collection {collection_name}: {e}")
                raise

    def add_paper_sections(self, paper: Paper, section_embeddings: Dict[str, np.ndarray]) -> Dict[str, str]:
        """
        Add paper sections to appropriate ChromaDB collections

        Args:
            paper: Paper object with structured sections
            section_embeddings: Dict mapping section types to embeddings

        Returns:
            Dict mapping section types to ChromaDB document IDs
        """
        if not paper.has_structured_sections:
            logger.warning(f"Paper {paper.pmid} has no structured sections")
            return {}

        document_ids = {}

        for section_type, section_data in paper.structured_sections.items():
            if section_type not in section_embeddings:
                logger.warning(f"No embedding found for section {section_type} in paper {paper.pmid}")
                continue

            try:
                # Generate unique document ID
                doc_id = f"{paper.clean_pmid}_{section_type}_{uuid.uuid4().hex[:8]}"

                # Prepare section content
                content = section_data.get('content', '')
                if not content.strip():
                    continue

                # Prepare rich metadata
                metadata = self._prepare_section_metadata(paper, section_type, section_data)

                # Add to appropriate collection
                collection = self.collections.get(section_type)
                if not collection:
                    # Fallback to abstracts collection for unknown sections
                    collection = self.collections.get('abstracts')
                    if not collection:
                        logger.error(f"No collection available for section {section_type}")
                        continue

                # Add document with embedding
                collection.add(
                    documents=[content],
                    embeddings=[section_embeddings[section_type].tolist()],
                    metadatas=[metadata],
                    ids=[doc_id]
                )

                document_ids[section_type] = doc_id

                # Update paper with ChromaDB ID
                paper.set_section_embedding_id(section_type, doc_id)

                logger.debug(f"✅ Added {section_type} section for paper {paper.pmid}")

            except Exception as e:
                logger.error(f"❌ Failed to add {section_type} section for paper {paper.pmid}: {e}")
                continue

        logger.info(f"📝 Added {len(document_ids)} sections to ChromaDB for paper {paper.pmid}")
        return document_ids

    def _prepare_section_metadata(self, paper: Paper, section_type: str, section_data: Dict) -> Dict[str, Any]:
        """Prepare rich metadata for section storage"""

        # Base paper metadata
        metadata = {
            "pmid": paper.clean_pmid,
            "pmcid": paper.clean_pmcid or "",
            "title": paper.title or "",
            "journal": paper.journal or "",
            "year": paper.year or 0,
            "doi": paper.doi or "",
            "authors": json.dumps(paper.authors) if paper.authors else "[]",
        }

        # Section-specific metadata
        metadata.update({
            "section_type": section_type,
            "content_length": section_data.get('content_length', 0),
            "confidence_score": section_data.get('confidence_score', 0.0),
            "has_citations": len(section_data.get('citations', [])) > 0,
            "has_figures": len(section_data.get('figures_tables', [])) > 0,
            "page_numbers": json.dumps(section_data.get('page_numbers', [])),
            "citation_count": len(section_data.get('citations', [])),
            "figure_table_count": len(section_data.get('figures_tables', []))
        })

        # Add research-specific metadata
        if section_type == 'methods':
            methodology = paper.get_research_methodology()
            metadata["research_methodology"] = methodology or "unknown"

        elif section_type == 'results':
            findings = paper.get_key_findings()
            metadata["finding_type"] = "significant" if findings else "descriptive"
            metadata["findings_count"] = len(findings)

        elif section_type in ['discussions', 'conclusions']:
            limitations = paper.get_limitations()
            metadata["limitation_mentioned"] = len(limitations) > 0
            metadata["limitations_count"] = len(limitations)

        # Quality indicators
        metadata.update({
            "parsing_quality": paper.section_parsing_quality,
            "structure_confidence": paper.structure_confidence,
            "paper_has_fulltext": paper.has_fulltext
        })

        return metadata

    def query_by_section(self,
                        query_embedding: np.ndarray,
                        section_types: List[str] = None,
                        filters: Dict[str, Any] = None,
                        n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Query specific section types with metadata filtering

        Args:
            query_embedding: Query vector embedding
            section_types: List of section types to search (default: all)
            filters: ChromaDB metadata filters
            n_results: Number of results to return

        Returns:
            List of search results with metadata
        """
        if section_types is None:
            section_types = list(self.collections.keys())

        all_results = []

        for section_type in section_types:
            collection = self.collections.get(section_type)
            if not collection:
                continue

            try:
                # Perform similarity search
                results = collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=min(n_results, collection.count()),
                    where=filters,
                    include=['documents', 'metadatas', 'distances']
                )

                # Process results
                for i in range(len(results['documents'][0])):
                    result = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'section_type': section_type,
                        'collection': collection.name
                    }
                    all_results.append(result)

            except Exception as e:
                logger.error(f"❌ Error querying {section_type} collection: {e}")
                continue

        # Sort by similarity score and return top results
        all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return all_results[:n_results]

    def query_cross_sections(self,
                           query_embedding: np.ndarray,
                           primary_sections: List[str],
                           secondary_sections: List[str],
                           filters: Dict[str, Any] = None,
                           n_results: int = 5) -> Dict[str, List[Dict]]:
        """
        Perform cross-section analysis (e.g., methods + results)

        Args:
            query_embedding: Query vector embedding
            primary_sections: Primary sections to search
            secondary_sections: Secondary sections for context
            filters: Metadata filters
            n_results: Results per section type

        Returns:
            Dict with results grouped by section type
        """
        results = {
            'primary': self.query_by_section(
                query_embedding, primary_sections, filters, n_results
            ),
            'secondary': self.query_by_section(
                query_embedding, secondary_sections, filters, n_results
            )
        }

        # Find papers that appear in both primary and secondary results
        primary_pmids = {r['metadata']['pmid'] for r in results['primary']}
        secondary_pmids = {r['metadata']['pmid'] for r in results['secondary']}
        common_papers = primary_pmids & secondary_pmids

        results['cross_section_papers'] = list(common_papers)
        results['cross_section_count'] = len(common_papers)

        return results

    def filter_by_research_context(self,
                                 query_embedding: np.ndarray,
                                 research_context: str,
                                 n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Filter results based on research context (methodology, findings, etc.)

        Args:
            query_embedding: Query vector embedding
            research_context: Type of research context to focus on
            n_results: Number of results

        Returns:
            Filtered and ranked results
        """
        context_mapping = {
            'methodology': ['methods'],
            'findings': ['results'],
            'background': ['introductions'],
            'analysis': ['discussions'],
            'limitations': ['discussions', 'conclusions'],
            'future_work': ['discussions', 'conclusions']
        }

        target_sections = context_mapping.get(research_context, list(self.collections.keys()))

        # Build context-specific filters
        filters = {}
        if research_context == 'methodology':
            filters = {"research_methodology": {"$ne": "unknown"}}
        elif research_context == 'findings':
            filters = {"finding_type": "significant"}
        elif research_context == 'limitations':
            filters = {"limitation_mentioned": True}

        return self.query_by_section(
            query_embedding=query_embedding,
            section_types=target_sections,
            filters=filters,
            n_results=n_results
        )

    def get_paper_sections(self, pmid: str) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve all sections for a specific paper

        Args:
            pmid: PubMed ID

        Returns:
            Dict mapping section types to section data
        """
        paper_sections = {}

        for section_type, collection in self.collections.items():
            try:
                results = collection.get(
                    where={"pmid": pmid},
                    include=['documents', 'metadatas']
                )

                if results['documents']:
                    paper_sections[section_type] = {
                        'content': results['documents'][0],
                        'metadata': results['metadatas'][0]
                    }

            except Exception as e:
                logger.error(f"❌ Error retrieving {section_type} for paper {pmid}: {e}")
                continue

        return paper_sections

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about all collections"""
        stats = {
            'total_collections': len(self.collections),
            'collection_details': {},
            'total_documents': 0
        }

        for section_type, collection in self.collections.items():
            try:
                count = collection.count()
                stats['collection_details'][section_type] = {
                    'name': collection.name,
                    'document_count': count,
                    'section_type': section_type
                }
                stats['total_documents'] += count

            except Exception as e:
                logger.error(f"❌ Error getting stats for {section_type}: {e}")
                stats['collection_details'][section_type] = {'error': str(e)}

        return stats

    def delete_paper(self, pmid: str) -> Dict[str, bool]:
        """
        Delete all sections of a paper from ChromaDB

        Args:
            pmid: PubMed ID

        Returns:
            Dict showing deletion status for each section type
        """
        deletion_status = {}

        for section_type, collection in self.collections.items():
            try:
                # Find documents for this paper
                results = collection.get(
                    where={"pmid": pmid},
                    include=['documents']
                )

                if results['ids']:
                    # Delete documents
                    collection.delete(ids=results['ids'])
                    deletion_status[section_type] = True
                    logger.debug(f"✅ Deleted {len(results['ids'])} {section_type} documents for paper {pmid}")
                else:
                    deletion_status[section_type] = True  # Nothing to delete

            except Exception as e:
                logger.error(f"❌ Error deleting {section_type} for paper {pmid}: {e}")
                deletion_status[section_type] = False

        return deletion_status

    def reset_collections(self) -> bool:
        """
        Reset all collections (delete all data)
        Use with caution!

        Returns:
            True if successful
        """
        try:
            for section_type, collection in self.collections.items():
                collection.delete()
                logger.info(f"🗑️ Reset collection: {section_type}")

            logger.info("🗑️ All collections reset successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Error resetting collections: {e}")
            return False

    def backup_metadata(self, backup_path: str) -> bool:
        """
        Backup collection metadata to JSON file

        Args:
            backup_path: Path for backup file

        Returns:
            True if successful
        """
        try:
            backup_data = {
                'collections': {},
                'stats': self.get_collection_stats(),
                'backup_timestamp': str(pd.Timestamp.now())
            }

            for section_type, collection in self.collections.items():
                try:
                    # Get all metadata (without embeddings for size efficiency)
                    results = collection.get(include=['metadatas'])
                    backup_data['collections'][section_type] = {
                        'metadata_list': results['metadatas'],
                        'document_count': len(results['metadatas'])
                    }
                except Exception as e:
                    backup_data['collections'][section_type] = {'error': str(e)}

            # Save to file
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)

            logger.info(f"💾 Backup saved to {backup_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            return False

    def search_by_methodology(self,
                            query_embedding: np.ndarray,
                            methodology_types: List[str],
                            n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search specifically for papers using certain methodologies

        Args:
            query_embedding: Query vector embedding
            methodology_types: List of methodology types to filter by
            n_results: Number of results

        Returns:
            Filtered results focusing on methodology
        """
        filters = {
            "research_methodology": {"$in": methodology_types}
        }

        return self.query_by_section(
            query_embedding=query_embedding,
            section_types=['methods'],
            filters=filters,
            n_results=n_results
        )

    def search_significant_findings(self,
                                  query_embedding: np.ndarray,
                                  min_findings_count: int = 1,
                                  n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for papers with significant findings

        Args:
            query_embedding: Query vector embedding
            min_findings_count: Minimum number of findings required
            n_results: Number of results

        Returns:
            Results from papers with significant findings
        """
        filters = {
            "$and": [
                {"finding_type": "significant"},
                {"findings_count": {"$gte": min_findings_count}}
            ]
        }

        return self.query_by_section(
            query_embedding=query_embedding,
            section_types=['results'],
            filters=filters,
            n_results=n_results
        )