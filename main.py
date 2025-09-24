#!/usr/bin/env python3
"""
Enhanced Scientific Literature Analysis Pipeline
Combines full-text PDF processing, RAG, vector search, and advanced analytics
"""

import os
import json
import time
import re
import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Tuple
import logging
from statistics import mode, StatisticsError

# Core libraries
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

# PDF processing
import fitz  # PyMuPDF

# NLP libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from textblob import TextBlob

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Vector search and embeddings
from sentence_transformers import SentenceTransformer
import faiss

# Network analysis
import networkx as nx

# LLM integrations
try:
    import ollama  # For local LLMs
except ImportError:
    ollama = None

try:
    from openai import OpenAI  # For OpenAI API
except ImportError:
    OpenAI = None

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


class EnhancedPubMedAnalyzer:
    """Enhanced analyzer with full-text processing and RAG capabilities"""

    def __init__(
        self,
        email: str,
        api_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        deepseek_key: Optional[str] = None,
    ):
        """
        Initialize the enhanced analyzer

        Args:
            email: Your email for NCBI API
            api_key: Optional NCBI API key for higher rate limits
            openai_key: Optional OpenAI API key for GPT models
            deepseek_key: Optional DeepSeek API key
        """
        self.email = email
        self.api_key = api_key
        self.openai_key = openai_key
        self.deepseek_key = deepseek_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

        # Data storage
        self.papers = []
        self.processed_papers = []
        self.full_texts = {}  # Store full-text content
        self.sections = {}  # Store sectioned content

        # Directories
        self.pdf_dir = "pdfs"
        self.sections_dir = "sections"
        self.index_dir = "vector_indices"
        os.makedirs(self.pdf_dir, exist_ok=True)
        os.makedirs(self.sections_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)

        # Initialize NLP tools
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        # Initialize sentence transformer for embeddings
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Vector indices
        self.abstract_index = None
        self.fulltext_index = None
        self.section_index = None

    def search_papers(
        self,
        query: str,
        max_results: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[str]:
        """
        Search PubMed for papers matching query
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "email": self.email,
        }

        if self.api_key:
            params["api_key"] = self.api_key

        if start_date and end_date:
            params["datetype"] = "pdat"
            params["mindate"] = start_date
            params["maxdate"] = end_date

        response = requests.get(f"{self.base_url}esearch.fcgi", params=params)
        data = response.json()

        pmids = data.get("esearchresult", {}).get("idlist", [])
        logger.info(f"Found {len(pmids)} papers for query: {query}")

        return pmids

    def fetch_papers_metadata(
        self, pmids: List[str], batch_size: int = 20
    ) -> List[Dict]:
        """
        Fetch paper metadata from PubMed
        """
        papers = []

        for i in tqdm(range(0, len(pmids), batch_size), desc="Fetching metadata"):
            batch = pmids[i : i + batch_size]
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "email": self.email,
            }

            if self.api_key:
                params["api_key"] = self.api_key

            response = requests.get(f"{self.base_url}efetch.fcgi", params=params)

            # Parse XML
            root = ET.fromstring(response.content)

            for article in root.findall(".//PubmedArticle"):
                paper = self._parse_article(article)
                if paper:
                    # Try to get PMC ID for full-text access
                    pmcid = self._get_pmcid(paper["pmid"])
                    if pmcid:
                        paper["pmcid"] = pmcid
                    papers.append(paper)

            # Rate limiting
            time.sleep(0.34 if not self.api_key else 0.1)

        self.papers = papers
        logger.info(f"Fetched metadata for {len(papers)} papers")
        return papers

    def _parse_article(self, article: ET.Element) -> Dict:
        """Parse XML article into dictionary"""
        paper = {}

        try:
            # Extract PMID
            pmid = article.find(".//PMID")
            paper["pmid"] = pmid.text if pmid is not None else ""

            # Extract title
            title = article.find(".//ArticleTitle")
            paper["title"] = title.text if title is not None else ""

            # Extract abstract
            abstract_texts = []
            for abstract in article.findall(".//Abstract/AbstractText"):
                if abstract.text:
                    abstract_texts.append(abstract.text)
            paper["abstract"] = " ".join(abstract_texts)

            # Extract authors
            authors = []
            for author in article.findall(".//Author"):
                last_name = author.find("LastName")
                fore_name = author.find("ForeName")
                if last_name is not None and fore_name is not None:
                    authors.append(f"{fore_name.text} {last_name.text}")
            paper["authors"] = authors

            # Extract publication date
            pub_date = article.find(".//PubDate")
            if pub_date is not None:
                year = pub_date.find("Year")
                month = pub_date.find("Month")
                paper["year"] = int(year.text) if year is not None else None
                paper["month"] = month.text if month is not None else None

            # Extract journal
            journal = article.find(".//Journal/Title")
            paper["journal"] = journal.text if journal is not None else ""

            # Extract DOI
            doi = None
            for aid in article.findall(".//ArticleId"):
                if aid.get("IdType") == "doi":
                    doi = aid.text
                    break
            paper["doi"] = doi

            # Extract keywords and MeSH terms
            keywords = []
            for keyword in article.findall(".//Keyword"):
                if keyword.text:
                    keywords.append(keyword.text)
            paper["keywords"] = keywords

            mesh_terms = []
            for mesh in article.findall(".//MeshHeading/DescriptorName"):
                if mesh.text:
                    mesh_terms.append(mesh.text)
            paper["mesh_terms"] = mesh_terms

            return paper

        except Exception as e:
            logger.error(f"Error parsing article: {e}")
            return None

    def _get_pmcid(self, pmid: str) -> Optional[str]:
        """Get PMC ID from PubMed ID for full-text access"""
        try:
            params = {
                "dbfrom": "pubmed",
                "db": "pmc",
                "id": pmid,
                "retmode": "json",
                "email": self.email,
            }
            response = requests.get(f"{self.base_url}elink.fcgi", params=params)
            data = response.json()

            link_sets = data.get("linksets", [])
            if link_sets and "linksetdbs" in link_sets[0]:
                for linksetdb in link_sets[0]["linksetdbs"]:
                    if linksetdb["dbto"] == "pmc":
                        return "PMC" + linksetdb["links"][0]
        except:
            pass
        return None

    async def _download_pdf(self, session: aiohttp.ClientSession, pmcid: str) -> bool:
        """Download a single PDF asynchronously"""
        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
        path = os.path.join(self.pdf_dir, f"{pmcid}.pdf")

        if os.path.exists(path):
            logger.debug(f"PDF already exists for {pmcid}")
            return True

        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    with open(path, "wb") as f:
                        f.write(content)
                    logger.debug(f"Downloaded PDF for {pmcid}")
                    return True
                else:
                    logger.warning(
                        f"Failed to download PDF for {pmcid}: Status {response.status}"
                    )
                    return False
        except Exception as e:
            logger.error(f"Error downloading PDF for {pmcid}: {e}")
            return False

    async def download_full_texts_async(self, pmcids: List[str]) -> Dict[str, bool]:
        """Download PDFs asynchronously for papers with PMC access"""
        logger.info(f"Starting async download of {len(pmcids)} PDFs")

        async with aiohttp.ClientSession() as session:
            tasks = [self._download_pdf(session, pmcid) for pmcid in pmcids]
            results = await asyncio.gather(*tasks)

        download_status = {pmcid: success for pmcid, success in zip(pmcids, results)}
        logger.info(f"Downloaded {sum(results)} PDFs successfully")
        return download_status

    def extract_sections_from_pdf(self, pdf_path: str) -> Dict[str, str]:
        """Extract structured sections from PDF"""
        sections = {}
        current_section = "Unknown"

        try:
            doc = fitz.open(pdf_path)
            all_blocks = []
            font_sizes = []

            # Extract all text blocks with metadata
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                font_sizes.append(span["size"])
                                all_blocks.append({
                                    "text": span["text"].strip(),
                                    "size": span["size"],
                                    "font": span["font"],
                                    "bold": bool(span["flags"] & 2),
                                    "italic": bool(span["flags"] & 1),
                                    "bbox": span["bbox"],
                                    "page": page_num,
                                })

            if not all_blocks:
                doc.close()
                return sections

            # Determine body font size
            try:
                body_size = mode(font_sizes)
            except StatisticsError:
                body_size = sorted(font_sizes)[len(font_sizes) // 2]

            # Sort blocks by page and position
            all_blocks.sort(key=lambda b: (b["page"], b["bbox"][1], b["bbox"][0]))

            # Known section patterns
            section_patterns = [
                r"^abstract$",
                r"^introduction$",
                r"^background$",
                r"^methods?$",
                r"^materials?\s+and\s+methods?$",
                r"^results?$",
                r"^discussion$",
                r"^conclusions?$",
                r"^references$",
                r"^acknowledgments?$",
                r"^supplementary",
                r"^appendix",
            ]

            current_text = ""
            for block in all_blocks:
                text = block["text"]
                if not text:
                    continue

                size = block["size"]
                is_bold = block["bold"]
                stripped_lower = text.lower().strip()

                # Check if this is a section header
                is_section = False
                if size > body_size * 1.1 or is_bold or text.isupper():
                    for pattern in section_patterns:
                        if re.match(pattern, stripped_lower):
                            is_section = True
                            break

                    # Also check numbered sections
                    if re.match(r"^\d+\.?\s+\w+", text):
                        is_section = True

                if is_section:
                    # Save previous section
                    if current_section and current_text.strip():
                        sections[current_section] = current_text.strip()
                    current_section = text
                    current_text = ""
                else:
                    current_text += text + " "

            # Save last section
            if current_section and current_text.strip():
                sections[current_section] = current_text.strip()

            doc.close()

        except Exception as e:
            logger.error(f"Error extracting sections from {pdf_path}: {e}")

        return sections

    def process_full_texts(self):
        """Process all downloaded PDFs and extract sections"""
        pmcids_with_pdfs = [p["pmcid"] for p in self.papers if "pmcid" in p]

        for pmcid in tqdm(pmcids_with_pdfs, desc="Processing PDFs"):
            pdf_path = os.path.join(self.pdf_dir, f"{pmcid}.pdf")
            if not os.path.exists(pdf_path):
                continue

            sections = self.extract_sections_from_pdf(pdf_path)
            if sections:
                self.sections[pmcid] = sections

                # Save sections to JSON
                json_path = os.path.join(self.sections_dir, f"{pmcid}.json")
                with open(json_path, "w") as f:
                    json.dump(sections, f)

                # Combine all sections for full text
                self.full_texts[pmcid] = " ".join(sections.values())

        logger.info(f"Processed {len(self.sections)} full-text PDFs")

    def build_vector_indices(self):
        """Build FAISS indices for abstracts, full texts, and sections"""
        indices = {}
        metadata = {}

        # Build abstract index
        if self.papers:
            abstracts = [p.get("abstract", "") for p in self.papers]
            abstracts = [a for a in abstracts if a]  # Filter empty

            if abstracts:
                logger.info("Building abstract index")
                abstract_embeddings = self.sentence_model.encode(abstracts)

                dim = abstract_embeddings.shape[1]
                abstract_index = faiss.IndexFlatL2(dim)
                abstract_index.add(abstract_embeddings.astype(np.float32))

                self.abstract_index = abstract_index
                indices["abstract"] = abstract_index
                metadata["abstract"] = [
                    {"pmid": p["pmid"], "title": p["title"]}
                    for p in self.papers
                    if p.get("abstract")
                ]

                # Save index
                faiss.write_index(
                    abstract_index, os.path.join(self.index_dir, "abstract_index.faiss")
                )

        # Build full-text index
        if self.full_texts:
            logger.info("Building full-text index")
            texts = list(self.full_texts.values())
            pmcids = list(self.full_texts.keys())

            # Chunk long texts for better retrieval
            chunked_texts = []
            chunked_metadata = []

            for pmcid, text in zip(pmcids, texts):
                # Split into chunks of ~1000 characters
                chunks = [text[i : i + 1000] for i in range(0, len(text), 800)]
                chunked_texts.extend(chunks)
                chunked_metadata.extend([
                    {"pmcid": pmcid, "chunk_idx": i} for i in range(len(chunks))
                ])

            fulltext_embeddings = self.sentence_model.encode(chunked_texts)

            dim = fulltext_embeddings.shape[1]
            fulltext_index = faiss.IndexFlatL2(dim)
            fulltext_index.add(fulltext_embeddings.astype(np.float32))

            self.fulltext_index = fulltext_index
            indices["fulltext"] = fulltext_index
            metadata["fulltext"] = chunked_metadata

            # Save index
            faiss.write_index(
                fulltext_index, os.path.join(self.index_dir, "fulltext_index.faiss")
            )

        # Build section index
        if self.sections:
            logger.info("Building section index")
            section_texts = []
            section_metadata = []

            for pmcid, sections in self.sections.items():
                for section_name, section_text in sections.items():
                    section_texts.append(section_text)
                    section_metadata.append({"pmcid": pmcid, "section": section_name})

            if section_texts:
                section_embeddings = self.sentence_model.encode(section_texts)

                dim = section_embeddings.shape[1]
                section_index = faiss.IndexFlatL2(dim)
                section_index.add(section_embeddings.astype(np.float32))

                self.section_index = section_index
                indices["section"] = section_index
                metadata["section"] = section_metadata

                # Save index
                faiss.write_index(
                    section_index, os.path.join(self.index_dir, "section_index.faiss")
                )

        # Save metadata
        with open(os.path.join(self.index_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        logger.info(f"Built {len(indices)} vector indices")
        return indices, metadata

    def semantic_search(
        self, query: str, index_type: str = "fulltext", top_k: int = 5
    ) -> List[Dict]:
        """Perform semantic search using FAISS indices"""
        if index_type == "abstract" and self.abstract_index is None:
            logger.warning("Abstract index not available")
            return []
        elif index_type == "fulltext" and self.fulltext_index is None:
            logger.warning("Fulltext index not available")
            return []
        elif index_type == "section" and self.section_index is None:
            logger.warning("Section index not available")
            return []

        # Load metadata
        with open(os.path.join(self.index_dir, "metadata.json"), "r") as f:
            all_metadata = json.load(f)

        # Encode query
        query_embedding = self.sentence_model.encode(query)

        # Search
        if index_type == "abstract":
            index = self.abstract_index
            metadata = all_metadata.get("abstract", [])
        elif index_type == "fulltext":
            index = self.fulltext_index
            metadata = all_metadata.get("fulltext", [])
        else:
            index = self.section_index
            metadata = all_metadata.get("section", [])

        distances, indices = index.search(
            np.array([query_embedding]).astype(np.float32), min(top_k, index.ntotal)
        )

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            result = metadata[idx].copy()
            result["distance"] = float(dist)
            result["relevance_score"] = 1 / (1 + float(dist))
            results.append(result)

        return results

    def rag_query(
        self,
        query: str,
        llm_provider: str = "openai",
        index_type: str = "fulltext",
        top_k: int = 5,
    ) -> Dict:
        """
        Perform RAG query with multiple LLM options

        Args:
            query: The question to answer
            llm_provider: 'openai', 'ollama', or 'deepseek'
            index_type: 'abstract', 'fulltext', or 'section'
            top_k: Number of relevant chunks to retrieve
        """
        # Retrieve relevant context
        search_results = self.semantic_search(query, index_type, top_k)

        if not search_results:
            return {
                "query": query,
                "response": "No relevant documents found.",
                "sources": [],
            }

        # Build context
        contexts = []
        sources = []

        for result in search_results:
            pmcid = result.get("pmcid")

            # Get the actual text based on index type
            if index_type == "abstract":
                paper = next((p for p in self.papers if p.get("pmcid") == pmcid), None)
                if paper:
                    text = paper.get("abstract", "")[:1000]
                    citation = f"{', '.join(paper['authors'][:3])} et al. ({paper.get('year')})"
                else:
                    continue
            elif index_type == "section":
                section_name = result.get("section")
                text = self.sections.get(pmcid, {}).get(section_name, "")[:1000]
                citation = f"PMC{pmcid}, Section: {section_name}"
            else:  # fulltext
                text = self.full_texts.get(pmcid, "")[:1000]
                citation = f"PMC{pmcid}"

            contexts.append(f"[Source: {citation}]\n{text}")
            sources.append({
                "pmcid": pmcid,
                "citation": citation,
                "relevance_score": result["relevance_score"],
            })

        # Build prompt
        prompt = f"""Answer the following question based on the provided scientific literature context.
        
Question: {query}

Context from scientific papers:
{chr(10).join(contexts)}

Please provide a comprehensive answer based on the context, citing the sources when making claims."""

        # Generate response using selected LLM
        response = self._generate_llm_response(prompt, llm_provider)

        return {
            "query": query,
            "response": response,
            "sources": sources,
            "provider": llm_provider,
        }

    def _generate_llm_response(self, prompt: str, provider: str = "openai") -> str:
        """Generate response using various LLM providers"""
        try:
            if provider == "openai" and self.openai_key and OpenAI:
                client = OpenAI(api_key=self.openai_key)
                response = client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1000,
                )
                return response.choices[0].message.content

            elif provider == "ollama" and ollama:
                response = ollama.chat(
                    model="llama2", messages=[{"role": "user", "content": prompt}]
                )
                return response["message"]["content"]

            elif provider == "deepseek" and self.deepseek_key:
                headers = {
                    "Authorization": f"Bearer {self.deepseek_key}",
                    "Content-Type": "application/json",
                }
                data = {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                }
                response = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                )
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    return f"API error: {response.status_code}"
            else:
                return "No LLM provider available. Please configure OpenAI, Ollama, or DeepSeek."

        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"Error generating response: {str(e)}"

    def comprehensive_analysis(self, save_outputs: bool = True) -> Dict:
        """Perform all analyses and save results"""
        results = {}

        # 1. Basic statistics
        stats = {
            "total_papers": len(self.papers),
            "papers_with_fulltext": len(self.full_texts),
            "papers_by_year": Counter([
                p.get("year") for p in self.papers if p.get("year")
            ]),
            "top_journals": Counter([
                p.get("journal") for p in self.papers
            ]).most_common(10),
            "top_keywords": Counter([
                kw for p in self.papers for kw in p.get("keywords", [])
            ]).most_common(20),
        }
        results["statistics"] = stats

        # 2. Topic modeling
        logger.info("Performing topic modeling")
        topics = self.topic_modeling(n_topics=10)
        results["topics"] = topics

        # 3. Collaboration network
        logger.info("Building collaboration network")
        collab_network = self.collaboration_network()
        results["collaboration_stats"] = {
            "total_authors": collab_network.number_of_nodes(),
            "total_collaborations": collab_network.number_of_edges(),
            "network_density": nx.density(collab_network)
            if collab_network.number_of_nodes() > 0
            else 0,
        }

        # 4. Research trends
        logger.info("Analyzing research trends")
        trends = self.trend_analysis()
        results["trends"] = trends

        # 5. Semantic clusters
        logger.info("Creating semantic clusters")
        clusters = self.semantic_similarity_clusters(n_clusters=5)
        results["clusters"] = clusters

        # 6. High-impact predictions
        logger.info("Predicting high-impact papers")
        predictions = self.citation_prediction()
        results["high_impact_predictions"] = predictions

        # 7. Research gaps
        logger.info("Identifying research gaps")
        gaps = self.research_gap_analysis()
        results["research_gaps"] = gaps

        # Save all results
        if save_outputs:
            with open("comprehensive_analysis.json", "w") as f:
                json.dump(results, f, indent=2, default=str)

            # Generate report
            self.generate_enhanced_report(results)

            # Create visualizations
            self.create_comprehensive_visualizations(results)

        return results

    def topic_modeling(self, n_topics: int = 10, use_fulltext: bool = True) -> Dict:
        """Enhanced topic modeling using full texts when available"""
        # Prepare texts
        if use_fulltext and self.full_texts:
            texts = list(self.full_texts.values())
            doc_ids = list(self.full_texts.keys())
        else:
            texts = [p.get("abstract", "") for p in self.papers if p.get("abstract")]
            doc_ids = [p["pmid"] for p in self.papers if p.get("abstract")]

        # Process texts
        processed_texts = []
        for text in texts:
            tokens = word_tokenize(text.lower())
            tokens = [
                self.lemmatizer.lemmatize(t)
                for t in tokens
                if t not in self.stop_words and len(t) > 2
            ]
            processed_texts.append(" ".join(tokens))

        # Vectorize
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        doc_term_matrix = vectorizer.fit_transform(processed_texts)

        # LDA
        lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        doc_topic_matrix = lda_model.fit_transform(doc_term_matrix)

        # Extract topics
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda_model.components_):
            top_indices = topic.argsort()[-15:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topics.append({
                "id": topic_idx,
                "words": top_words,
                "weights": topic[top_indices].tolist(),
            })

        return {
            "topics": topics,
            "doc_topic_assignments": {
                doc_id: doc_topic_matrix[i].argmax() for i, doc_id in enumerate(doc_ids)
            },
        }

    def collaboration_network(self) -> nx.Graph:
        """Build enhanced collaboration network with metrics"""
        G = nx.Graph()

        for paper in self.papers:
            authors = paper.get("authors", [])

            # Add nodes with attributes
            for author in authors:
                if not G.has_node(author):
                    G.add_node(author, papers=1, years=set([paper.get("year")]))
                else:
                    G.nodes[author]["papers"] += 1
                    G.nodes[author]["years"].add(paper.get("year"))

            # Add edges between co-authors
            for i, author1 in enumerate(authors):
                for author2 in authors[i + 1 :]:
                    if G.has_edge(author1, author2):
                        G[author1][author2]["weight"] += 1
                        G[author1][author2]["papers"].append(paper["pmid"])
                    else:
                        G.add_edge(author1, author2, weight=1, papers=[paper["pmid"]])

        return G

    def trend_analysis(self) -> Dict:
        """Enhanced trend analysis with multiple metrics"""
        df = pd.DataFrame(self.papers)

        if "year" not in df.columns or df["year"].isna().all():
            return {}

        df = df[df["year"].notna()]

        # Publication trends
        yearly_counts = df.groupby("year").size()

        # Keyword evolution
        keyword_trends = defaultdict(lambda: defaultdict(int))
        for _, row in df.iterrows():
            year = row["year"]
            for keyword in row.get("keywords", []):
                keyword_trends[year][keyword] += 1

        # MeSH term trends
        mesh_trends = defaultdict(lambda: defaultdict(int))
        for _, row in df.iterrows():
            year = row["year"]
            for mesh in row.get("mesh_terms", []):
                mesh_trends[year][mesh] += 1

        # Sentiment analysis on abstracts
        sentiments = []
        for abstract in df["abstract"]:
            if abstract:
                blob = TextBlob(abstract)
                sentiments.append(blob.sentiment.polarity)
            else:
                sentiments.append(0)

        df["sentiment"] = sentiments
        sentiment_trends = df.groupby("year")["sentiment"].mean()

        # Author productivity trends
        author_counts = df.groupby("year").apply(
            lambda x: len(
                set([author for authors in x["authors"] for author in authors])
            )
        )

        return {
            "yearly_publications": yearly_counts.to_dict(),
            "keyword_trends": dict(keyword_trends),
            "mesh_trends": dict(mesh_trends),
            "sentiment_trends": sentiment_trends.to_dict(),
            "author_counts": author_counts.to_dict(),
        }

    def semantic_similarity_clusters(self, n_clusters: int = 5) -> Dict:
        """Enhanced clustering with multiple text sources"""
        # Use full texts if available, otherwise abstracts
        if self.full_texts:
            texts = list(self.full_texts.values())
            doc_ids = list(self.full_texts.keys())
        else:
            texts = [p.get("abstract", "") for p in self.papers if p.get("abstract")]
            doc_ids = [p["pmid"] for p in self.papers if p.get("abstract")]

        if not texts:
            return {}

        # Vectorize
        vectorizer = TfidfVectorizer(max_features=500)
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(texts)), random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)

        # Organize results
        clustered_docs = defaultdict(list)
        for idx, cluster_id in enumerate(clusters):
            doc_id = doc_ids[idx]
            paper = next(
                (
                    p
                    for p in self.papers
                    if p.get("pmid") == doc_id or p.get("pmcid") == doc_id
                ),
                {},
            )
            clustered_docs[int(cluster_id)].append({
                "id": doc_id,
                "title": paper.get("title", "Unknown"),
                "year": paper.get("year"),
            })

        # Find cluster themes using top terms
        feature_names = vectorizer.get_feature_names_out()
        cluster_themes = {}

        for cluster_id in range(n_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            if len(cluster_indices) > 0:
                cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A[0]
                top_indices = cluster_tfidf.argsort()[-10:][::-1]
                top_terms = [feature_names[i] for i in top_indices]
                cluster_themes[cluster_id] = top_terms

        return {"clusters": dict(clustered_docs), "themes": cluster_themes}

    def citation_prediction(self) -> List[Dict]:
        """Enhanced citation prediction using multiple features"""
        predictions = []

        for paper in self.papers:
            # Calculate impact features
            features = {
                "num_authors": len(paper.get("authors", [])),
                "abstract_length": len(paper.get("abstract", "")),
                "num_keywords": len(paper.get("keywords", [])),
                "num_mesh": len(paper.get("mesh_terms", [])),
                "has_fulltext": paper.get("pmcid") in self.full_texts,
                "year_recent": max(0, 2025 - paper.get("year", 2020)),
            }

            # Add text complexity features if abstract exists
            if paper.get("abstract"):
                blob = TextBlob(paper["abstract"])
                features["sentiment_neutrality"] = 1 - abs(blob.sentiment.polarity)
                features["subjectivity"] = blob.sentiment.subjectivity
                features["sentence_count"] = len(blob.sentences)

            # Calculate composite score
            impact_score = (
                features.get("num_authors", 0) * 0.15
                + min(features.get("abstract_length", 0) / 1000, 1) * 0.20
                + features.get("num_keywords", 0) * 0.15
                + features.get("num_mesh", 0) * 0.10
                + features.get("sentiment_neutrality", 0) * 0.10
                + (1 - features.get("subjectivity", 1)) * 0.10
                + features.get("has_fulltext", 0) * 0.10
                + min(features.get("year_recent", 0) / 5, 1) * 0.10
            )

            predictions.append({
                "pmid": paper["pmid"],
                "title": paper["title"],
                "year": paper.get("year"),
                "impact_score": impact_score,
                "features": features,
            })

        # Sort by impact score
        predictions.sort(key=lambda x: x["impact_score"], reverse=True)

        return predictions[:20]  # Return top 20

    def research_gap_analysis(self) -> List[str]:
        """Enhanced research gap identification"""
        gaps = []

        # Extract all concepts
        all_concepts = set()
        concept_papers = defaultdict(set)

        for paper in self.papers:
            paper_concepts = set()

            # Add keywords and MeSH terms
            paper_concepts.update(paper.get("keywords", []))
            paper_concepts.update(paper.get("mesh_terms", []))

            # Add key entities from abstract
            if paper.get("abstract"):
                doc = self.nlp(paper["abstract"][:100000])
                paper_concepts.update([
                    ent.text
                    for ent in doc.ents
                    if ent.label_ in ["ORG", "PRODUCT", "DISEASE"]
                ])

            all_concepts.update(paper_concepts)
            for concept in paper_concepts:
                concept_papers[concept].add(paper["pmid"])

        # Find rarely co-occurring concepts
        concept_pairs = defaultdict(int)
        for paper in self.papers:
            concepts = set()
            concepts.update(paper.get("keywords", []))
            concepts.update(paper.get("mesh_terms", []))

            concepts_list = list(concepts)
            for i, c1 in enumerate(concepts_list):
                for c2 in concepts_list[i + 1 :]:
                    if c1 < c2:
                        concept_pairs[(c1, c2)] += 1

        # Identify gaps based on rare co-occurrences with high individual frequencies
        for (c1, c2), count in concept_pairs.items():
            freq1 = len(concept_papers[c1])
            freq2 = len(concept_papers[c2])

            # Both concepts are reasonably common but rarely appear together
            if freq1 >= 3 and freq2 >= 3 and count <= 1:
                gaps.append(
                    f"Potential research gap: Intersection of '{c1}' (appears in {freq1} papers) "
                    f"and '{c2}' (appears in {freq2} papers) - only {count} co-occurrence(s)"
                )

        # Sort by potential impact
        gaps.sort(
            key=lambda x: sum(int(n) for n in re.findall(r"\d+", x)), reverse=True
        )

        return gaps[:15]

    def generate_enhanced_report(self, results: Dict):
        """Generate comprehensive markdown report"""
        report = []
        report.append("# Enhanced Scientific Literature Analysis Report")
        report.append(
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report.append(
            f"\n**Analysis Pipeline:** PubMed + Full-Text + RAG + Advanced Analytics"
        )

        # Statistics
        stats = results.get("statistics", {})
        report.append(f"\n## Overview")
        report.append(f"- **Total papers analyzed:** {stats.get('total_papers', 0)}")
        report.append(
            f"- **Papers with full-text:** {stats.get('papers_with_fulltext', 0)}"
        )
        report.append(
            f"- **Coverage:** {stats.get('papers_with_fulltext', 0) / max(stats.get('total_papers', 1), 1) * 100:.1f}%"
        )

        # Top journals
        report.append(f"\n### Top Journals")
        for journal, count in stats.get("top_journals", [])[:5]:
            report.append(f"- {journal}: {count} papers")

        # Topics
        report.append(f"\n## Discovered Research Topics")
        topics_data = results.get("topics", {})
        for topic in topics_data.get("topics", [])[:5]:
            top_words = ", ".join(topic["words"][:8])
            report.append(f"\n**Topic {topic['id']}:** {top_words}")

        # Trends
        trends = results.get("trends", {})
        if trends:
            report.append(f"\n## Temporal Trends")

            yearly_pubs = trends.get("yearly_publications", {})
            if yearly_pubs:
                recent_years = sorted(yearly_pubs.keys())[-5:]
                report.append(f"\n### Recent Publication Trend")
                for year in recent_years:
                    report.append(f"- {year}: {yearly_pubs.get(year, 0)} papers")

            # Emerging keywords
            keyword_trends = trends.get("keyword_trends", {})
            if keyword_trends:
                recent_year = max(keyword_trends.keys()) if keyword_trends else None
                if recent_year and recent_year in keyword_trends:
                    report.append(f"\n### Emerging Keywords ({recent_year})")
                    top_keywords = sorted(
                        keyword_trends[recent_year].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:5]
                    for keyword, count in top_keywords:
                        report.append(f"- {keyword}: {count} occurrences")

        # High-impact predictions
        predictions = results.get("high_impact_predictions", [])
        if predictions:
            report.append(f"\n## Predicted High-Impact Papers")
            for paper in predictions[:5]:
                report.append(f"\n**{paper['title']}**")
                report.append(f"- Year: {paper.get('year', 'N/A')}")
                report.append(f"- Impact Score: {paper['impact_score']:.3f}")

        # Research gaps
        gaps = results.get("research_gaps", [])
        if gaps:
            report.append(f"\n## Identified Research Gaps")
            for gap in gaps[:5]:
                report.append(f"\n- {gap}")

        # Collaboration insights
        collab_stats = results.get("collaboration_stats", {})
        if collab_stats:
            report.append(f"\n## Collaboration Network")
            report.append(
                f"- Total researchers: {collab_stats.get('total_authors', 0)}"
            )
            report.append(
                f"- Total collaborations: {collab_stats.get('total_collaborations', 0)}"
            )
            report.append(
                f"- Network density: {collab_stats.get('network_density', 0):.4f}"
            )

        # Semantic clusters
        clusters_data = results.get("clusters", {})
        if clusters_data:
            report.append(f"\n## Semantic Research Clusters")
            themes = clusters_data.get("themes", {})
            clusters = clusters_data.get("clusters", {})

            for cluster_id, theme_words in themes.items():
                if cluster_id in clusters:
                    report.append(
                        f"\n**Cluster {cluster_id}** ({len(clusters[cluster_id])} papers)"
                    )
                    report.append(f"- Theme: {', '.join(theme_words[:5])}")

        # Save report
        with open("enhanced_analysis_report.md", "w") as f:
            f.write("\n".join(report))

        logger.info("Enhanced report generated: enhanced_analysis_report.md")

    def create_comprehensive_visualizations(self, results: Dict):
        """Create enhanced visualizations"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

        # 1. Publication trends
        trends = results.get("trends", {})
        yearly_pubs = trends.get("yearly_publications", {})
        if yearly_pubs:
            years = sorted(yearly_pubs.keys())
            counts = [yearly_pubs[y] for y in years]
            axes[0, 0].plot(years, counts, marker="o", linewidth=2, markersize=8)
            axes[0, 0].fill_between(years, counts, alpha=0.3)
            axes[0, 0].set_title(
                "Publication Trends Over Time", fontsize=12, fontweight="bold"
            )
            axes[0, 0].set_xlabel("Year")
            axes[0, 0].set_ylabel("Number of Papers")
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Top keywords
        stats = results.get("statistics", {})
        top_keywords = stats.get("top_keywords", [])[:10]
        if top_keywords:
            keywords, counts = zip(*top_keywords)
            axes[0, 1].barh(range(len(keywords)), counts, color="steelblue")
            axes[0, 1].set_yticks(range(len(keywords)))
            axes[0, 1].set_yticklabels(keywords)
            axes[0, 1].set_title("Top Keywords", fontsize=12, fontweight="bold")
            axes[0, 1].set_xlabel("Frequency")

        # 3. Journal distribution
        top_journals = stats.get("top_journals", [])[:8]
        if top_journals:
            journals, counts = zip(*top_journals)
            journals = [j[:30] + "..." if len(j) > 30 else j for j in journals]
            axes[0, 2].pie(counts, labels=journals, autopct="%1.1f%%")
            axes[0, 2].set_title("Journal Distribution", fontsize=12, fontweight="bold")

        # 4. Sentiment trends
        sentiment_trends = trends.get("sentiment_trends", {})
        if sentiment_trends:
            years = sorted(sentiment_trends.keys())
            sentiments = [sentiment_trends[y] for y in years]
            axes[1, 0].plot(years, sentiments, color="green", marker="s")
            axes[1, 0].axhline(y=0, color="red", linestyle="--", alpha=0.5)
            axes[1, 0].set_title(
                "Abstract Sentiment Over Time", fontsize=12, fontweight="bold"
            )
            axes[1, 0].set_xlabel("Year")
            axes[1, 0].set_ylabel("Average Sentiment")
            axes[1, 0].grid(True, alpha=0.3)

        # 5. Topic distribution
        topics_data = results.get("topics", {})
        if topics_data and "doc_topic_assignments" in topics_data:
            topic_counts = Counter(topics_data["doc_topic_assignments"].values())
            axes[1, 1].bar(topic_counts.keys(), topic_counts.values(), color="coral")
            axes[1, 1].set_title(
                "Document Distribution Across Topics", fontsize=12, fontweight="bold"
            )
            axes[1, 1].set_xlabel("Topic ID")
            axes[1, 1].set_ylabel("Number of Documents")

        # 6. Impact score distribution
        predictions = results.get("high_impact_predictions", [])
        if predictions:
            scores = [p["impact_score"] for p in predictions]
            axes[1, 2].hist(
                scores, bins=20, edgecolor="black", alpha=0.7, color="purple"
            )
            axes[1, 2].set_title(
                "Citation Impact Score Distribution", fontsize=12, fontweight="bold"
            )
            axes[1, 2].set_xlabel("Impact Score")
            axes[1, 2].set_ylabel("Frequency")

        # 7. Author productivity
        author_counts = trends.get("author_counts", {})
        if author_counts:
            years = sorted(author_counts.keys())
            counts = [author_counts[y] for y in years]
            axes[2, 0].bar(years, counts, color="teal")
            axes[2, 0].set_title(
                "Unique Authors per Year", fontsize=12, fontweight="bold"
            )
            axes[2, 0].set_xlabel("Year")
            axes[2, 0].set_ylabel("Number of Authors")
            axes[2, 0].tick_params(axis="x", rotation=45)

        # 8. Cluster sizes
        clusters_data = results.get("clusters", {})
        if clusters_data and "clusters" in clusters_data:
            clusters = clusters_data["clusters"]
            cluster_sizes = [len(papers) for papers in clusters.values()]
            axes[2, 1].pie(
                cluster_sizes,
                labels=[f"Cluster {i}" for i in range(len(cluster_sizes))],
                autopct="%1.1f%%",
                startangle=90,
            )
            axes[2, 1].set_title(
                "Semantic Cluster Distribution", fontsize=12, fontweight="bold"
            )

        # 9. Full-text vs Abstract coverage
        total = stats.get("total_papers", 0)
        fulltext = stats.get("papers_with_fulltext", 0)
        abstract_only = total - fulltext
        if total > 0:
            axes[2, 2].pie(
                [fulltext, abstract_only],
                labels=["Full-text", "Abstract only"],
                autopct="%1.1f%%",
                colors=["lightgreen", "lightcoral"],
            )
            axes[2, 2].set_title("Data Coverage", fontsize=12, fontweight="bold")

        plt.suptitle(
            "Enhanced Literature Analysis Dashboard",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig("enhanced_analysis_dashboard.png", dpi=300, bbox_inches="tight")
        plt.show()

        logger.info("Visualizations saved: enhanced_analysis_dashboard.png")


def main():
    """Main execution function with example usage"""

    # Configuration
    EMAIL = "your.email@example.com"  # Replace with your email
    NCBI_API_KEY = None  # Optional: Get from NCBI for higher rate limits
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")  # For GPT models
    DEEPSEEK_KEY = os.getenv(
        "DEEPSEEK_API_KEY"
    )  # For DeepSeek (very cheap alternative)

    # Search parameters
    SEARCH_QUERY = "machine learning AND medical diagnosis"
    MAX_PAPERS = 50
    START_DATE = "2020/01/01"
    END_DATE = "2024/12/31"

    # Initialize enhanced analyzer
    analyzer = EnhancedPubMedAnalyzer(
        email=EMAIL,
        api_key=NCBI_API_KEY,
        openai_key=OPENAI_KEY,
        deepseek_key=DEEPSEEK_KEY,
    )

    logger.info("Starting enhanced literature analysis pipeline...")

    # 1. Search for papers
    pmids = analyzer.search_papers(
        query=SEARCH_QUERY,
        max_results=MAX_PAPERS,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    if not pmids:
        logger.error("No papers found. Please adjust your search query.")
        return

    # 2. Fetch metadata
    papers = analyzer.fetch_papers_metadata(pmids)

    # 3. Download full-text PDFs (async)
    pmcids = [p["pmcid"] for p in papers if "pmcid" in p]
    if pmcids:
        logger.info(f"Downloading {len(pmcids)} full-text PDFs...")
        download_status = asyncio.run(analyzer.download_full_texts_async(pmcids))
        logger.info(f"Successfully downloaded {sum(download_status.values())} PDFs")

    # 4. Process full texts and extract sections
    analyzer.process_full_texts()

    # 5. Build vector indices for semantic search
    indices, metadata = analyzer.build_vector_indices()

    # 6. Example semantic search
    if analyzer.fulltext_index:
        logger.info("Performing example semantic search...")
        search_results = analyzer.semantic_search(
            "deep learning diagnosis accuracy", index_type="fulltext", top_k=5
        )
        logger.info(f"Found {len(search_results)} relevant documents")

    # 7. Example RAG query
    if OPENAI_KEY or DEEPSEEK_KEY:
        logger.info("Performing example RAG query...")
        rag_result = analyzer.rag_query(
            "What are the latest advances in using machine learning for medical diagnosis?",
            llm_provider="openai" if OPENAI_KEY else "deepseek",
            index_type="fulltext" if analyzer.fulltext_index else "abstract",
            top_k=5,
        )

        # Save RAG result
        with open("rag_example.json", "w") as f:
            json.dump(rag_result, f, indent=2)
        logger.info("RAG result saved to rag_example.json")

    # 8. Perform comprehensive analysis
    logger.info("Running comprehensive analysis suite...")
    results = analyzer.comprehensive_analysis(save_outputs=True)

    logger.info("\n" + "=" * 50)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("=" * 50)
    logger.info("\nGenerated outputs:")
    logger.info("- enhanced_analysis_report.md (comprehensive report)")
    logger.info("- enhanced_analysis_dashboard.png (visualization dashboard)")
    logger.info("- comprehensive_analysis.json (all analysis results)")
    logger.info("- pdfs/ (downloaded full-text PDFs)")
    logger.info("- sections/ (extracted paper sections)")
    logger.info("- vector_indices/ (FAISS indices for semantic search)")

    if OPENAI_KEY or DEEPSEEK_KEY:
        logger.info("- rag_example.json (example RAG query result)")

    logger.info(
        "\nYou can now use the semantic search and RAG capabilities for advanced queries!"
    )


if __name__ == "__main__":
    main()
