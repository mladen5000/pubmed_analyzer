#!/usr/bin/env python3
"""
Advanced NLP/ML Analysis Module
Enhanced natural language processing and machine learning capabilities for scientific literature
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from textblob import TextBlob
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import faiss
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class AdvancedNLPAnalyzer:
    """Advanced NLP and ML analysis for scientific literature"""

    def __init__(self):
        self.nlp = None
        self.sentence_model = None
        self.tfidf = None
        self.scaler = StandardScaler()
        self._load_models()

    def _load_models(self):
        """Lazy load NLP models"""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ Loaded spaCy model")

            # Load sentence transformer
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Loaded sentence transformer model")

        except Exception as e:
            logger.warning(f"⚠️ Model loading issue: {e}")

    def advanced_topic_modeling(self, texts: List[str], n_topics: int = 10) -> Dict[str, Any]:
        """Enhanced topic modeling with coherence scoring"""
        logger.info(f"🎯 Running advanced topic modeling ({n_topics} topics)")

        # Clean and preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts if text.strip()]

        if len(processed_texts) < 5:
            logger.warning("⚠️ Too few texts for reliable topic modeling")
            return {"error": "Insufficient text data"}

        # TF-IDF vectorization with advanced parameters
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )

        doc_term_matrix = vectorizer.fit_transform(processed_texts)
        feature_names = vectorizer.get_feature_names_out()

        # LDA with hyperparameter tuning
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            learning_decay=0.7,
            learning_offset=50.0,
            max_iter=100
        )

        doc_topic_matrix = lda.fit_transform(doc_term_matrix)

        # Extract topics with scores
        topics = {}
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-10:][::-1]
            topic_words = [(feature_names[i], topic[i]) for i in top_indices]
            topics[f"topic_{topic_idx}"] = {
                "words": topic_words,
                "coherence": float(np.mean([score for _, score in topic_words[:5]]))
            }

        # Document topic assignments
        doc_topics = []
        for i, doc_dist in enumerate(doc_topic_matrix):
            dominant_topic = np.argmax(doc_dist)
            confidence = float(np.max(doc_dist))
            doc_topics.append({
                "document_index": i,
                "dominant_topic": int(dominant_topic),
                "confidence": confidence,
                "topic_distribution": doc_dist.tolist()
            })

        return {
            "topics": topics,
            "document_topics": doc_topics,
            "n_topics": n_topics,
            "perplexity": float(lda.perplexity(doc_term_matrix))
        }

    def enhanced_sentiment_analysis(self, texts: List[str]) -> Dict[str, Any]:
        """Multi-level sentiment analysis with aspect detection"""
        logger.info("😊 Running enhanced sentiment analysis")

        results = {
            "overall_sentiment": {},
            "aspect_sentiments": {},
            "emotion_analysis": {},
            "document_sentiments": []
        }

        sentiments = []
        emotions = []
        aspects = ["method", "result", "conclusion", "limitation", "future work"]
        aspect_sentiments = {aspect: [] for aspect in aspects}

        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue

            # TextBlob sentiment
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            sentiments.append({
                "index": i,
                "polarity": polarity,
                "subjectivity": subjectivity,
                "label": self._classify_sentiment(polarity)
            })

            # Emotion detection using spaCy
            if self.nlp:
                doc = self.nlp(text.lower())
                emotion_keywords = self._extract_emotion_keywords(doc)
                emotions.append(emotion_keywords)

            # Aspect-based sentiment
            for aspect in aspects:
                aspect_sentiment = self._aspect_sentiment(text, aspect)
                if aspect_sentiment is not None:
                    aspect_sentiments[aspect].append(aspect_sentiment)

        # Aggregate results
        if sentiments:
            polarities = [s["polarity"] for s in sentiments]
            subjectivities = [s["subjectivity"] for s in sentiments]

            results["overall_sentiment"] = {
                "mean_polarity": float(np.mean(polarities)),
                "std_polarity": float(np.std(polarities)),
                "mean_subjectivity": float(np.mean(subjectivities)),
                "positive_ratio": len([p for p in polarities if p > 0.1]) / len(polarities),
                "negative_ratio": len([p for p in polarities if p < -0.1]) / len(polarities),
                "neutral_ratio": len([p for p in polarities if -0.1 <= p <= 0.1]) / len(polarities)
            }

        # Aspect sentiment aggregation
        for aspect, scores in aspect_sentiments.items():
            if scores:
                results["aspect_sentiments"][aspect] = {
                    "mean_sentiment": float(np.mean(scores)),
                    "count": len(scores)
                }

        results["document_sentiments"] = sentiments

        return results

    def advanced_clustering(self, texts: List[str], method: str = "kmeans") -> Dict[str, Any]:
        """Advanced clustering with multiple algorithms and evaluation"""
        logger.info(f"🎯 Running advanced clustering ({method})")

        if not texts or len(texts) < 3:
            return {"error": "Insufficient data for clustering"}

        # Generate embeddings
        embeddings = self.sentence_model.encode(texts)

        # Dimensionality reduction
        if embeddings.shape[1] > 50:
            pca = PCA(n_components=50)
            embeddings = pca.fit_transform(embeddings)

        results = {}

        if method == "kmeans":
            # K-means with elbow method
            best_k = self._find_optimal_clusters(embeddings, max_k=min(10, len(texts)//2))
            kmeans = KMeans(n_clusters=best_k, random_state=42)
            labels = kmeans.fit_predict(embeddings)

            results = {
                "method": "kmeans",
                "n_clusters": best_k,
                "labels": labels.tolist(),
                "silhouette_score": float(silhouette_score(embeddings, labels))
            }

        elif method == "dbscan":
            # DBSCAN with parameter optimization
            eps = self._find_optimal_eps(embeddings)
            dbscan = DBSCAN(eps=eps, min_samples=2)
            labels = dbscan.fit_predict(embeddings)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            results = {
                "method": "dbscan",
                "eps": eps,
                "n_clusters": n_clusters,
                "labels": labels.tolist(),
                "noise_points": int(np.sum(labels == -1))
            }

        # Add cluster summaries
        cluster_summaries = self._generate_cluster_summaries(texts, results["labels"])
        results["cluster_summaries"] = cluster_summaries

        return results

    def named_entity_recognition(self, texts: List[str]) -> Dict[str, Any]:
        """Enhanced NER for scientific literature"""
        logger.info("🔍 Running named entity recognition")

        if not self.nlp:
            return {"error": "spaCy model not available"}

        entities_by_type = {}
        all_entities = []

        for i, text in enumerate(texts):
            if not text:
                continue

            doc = self.nlp(text)
            doc_entities = []

            for ent in doc.ents:
                entity_info = {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": float(getattr(ent._, "confidence", 1.0))
                }
                doc_entities.append(entity_info)

                # Group by type
                if ent.label_ not in entities_by_type:
                    entities_by_type[ent.label_] = []
                entities_by_type[ent.label_].append(ent.text)

            all_entities.append({
                "document_index": i,
                "entities": doc_entities
            })

        # Count and rank entities
        entity_counts = {}
        for label, entities in entities_by_type.items():
            entity_counts[label] = Counter(entities).most_common(10)

        return {
            "entities_by_document": all_entities,
            "entities_by_type": entity_counts,
            "entity_types": list(entities_by_type.keys()),
            "total_entities": sum(len(doc["entities"]) for doc in all_entities)
        }

    def research_trend_analysis(self, papers_data: List[Dict]) -> Dict[str, Any]:
        """Advanced trend analysis with temporal patterns"""
        logger.info("📈 Running research trend analysis")

        df = pd.DataFrame(papers_data)
        if 'year' not in df.columns or df.empty:
            return {"error": "No temporal data available"}

        # Clean year data
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['year'])
        df = df[df['year'] >= 1990]  # Filter reasonable years

        results = {}

        # Publication trends
        year_counts = df['year'].value_counts().sort_index()
        results["publication_trends"] = {
            "yearly_counts": year_counts.to_dict(),
            "trend_slope": float(np.polyfit(year_counts.index, year_counts.values, 1)[0]),
            "peak_year": int(year_counts.idxmax())
        }

        # Journal trends
        if 'journal' in df.columns:
            journal_trends = {}
            for journal in df['journal'].value_counts().head(10).index:
                journal_data = df[df['journal'] == journal]['year'].value_counts().sort_index()
                if len(journal_data) > 1:
                    slope = float(np.polyfit(journal_data.index, journal_data.values, 1)[0])
                    journal_trends[journal] = {
                        "slope": slope,
                        "recent_count": int(journal_data.get(journal_data.index[-1], 0))
                    }
            results["journal_trends"] = journal_trends

        # Keyword evolution
        if 'keywords' in df.columns:
            keyword_evolution = self._analyze_keyword_evolution(df)
            results["keyword_evolution"] = keyword_evolution

        return results

    def citation_network_analysis(self, papers_data: List[Dict]) -> Dict[str, Any]:
        """Build and analyze citation networks"""
        logger.info("🕸️ Running citation network analysis")

        # Create citation graph
        G = nx.DiGraph()

        # Add nodes and edges (simplified - would need actual citation data)
        for i, paper in enumerate(papers_data):
            G.add_node(i,
                      title=paper.get('title', ''),
                      year=paper.get('year'),
                      journal=paper.get('journal', ''))

        # Basic network metrics (without actual citation links)
        if G.number_of_nodes() > 0:
            results = {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "density": nx.density(G),
                "is_connected": nx.is_connected(G.to_undirected()),
            }

            # Add centrality measures if there are edges
            if G.number_of_edges() > 0:
                degree_centrality = nx.degree_centrality(G)
                results["top_central_papers"] = sorted(
                    [(i, score) for i, score in degree_centrality.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:10]

            return results

        return {"error": "No network data available"}

    def research_gap_identification(self, texts: List[str], topics: Dict) -> List[Dict]:
        """Identify potential research gaps using topic modeling and clustering"""
        logger.info("🔍 Identifying research gaps")

        gaps = []

        # Analyze topic coverage
        if "topics" in topics:
            topic_strengths = {}
            for topic_id, topic_data in topics["topics"].items():
                coherence = topic_data.get("coherence", 0)
                topic_strengths[topic_id] = coherence

            # Identify weak topics as potential gaps
            weak_topics = sorted(topic_strengths.items(), key=lambda x: x[1])[:3]

            for topic_id, strength in weak_topics:
                topic_words = [word for word, _ in topics["topics"][topic_id]["words"][:5]]
                gaps.append({
                    "type": "weak_topic_coverage",
                    "topic_id": topic_id,
                    "strength": strength,
                    "keywords": topic_words,
                    "description": f"Limited research in: {', '.join(topic_words[:3])}"
                })

        # Identify methodology gaps
        method_keywords = ["method", "methodology", "approach", "technique", "algorithm"]
        method_mentions = sum(1 for text in texts for keyword in method_keywords if keyword in text.lower())

        if method_mentions / len(texts) < 0.3:  # Less than 30% mention methods
            gaps.append({
                "type": "methodology_gap",
                "description": "Limited discussion of research methodologies",
                "severity": "medium"
            })

        return gaps

    def _preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing"""
        if not text:
            return ""

        # Remove special characters, normalize whitespace
        import re
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()

    def _classify_sentiment(self, polarity: float) -> str:
        """Classify sentiment polarity"""
        if polarity > 0.1:
            return "positive"
        elif polarity < -0.1:
            return "negative"
        else:
            return "neutral"

    def _extract_emotion_keywords(self, doc) -> Dict[str, int]:
        """Extract emotion-related keywords using spaCy"""
        emotion_words = {
            "positive": ["good", "excellent", "effective", "successful", "promising"],
            "negative": ["poor", "failed", "limited", "difficult", "challenging"],
            "neutral": ["shown", "observed", "found", "reported", "described"]
        }

        emotions = {"positive": 0, "negative": 0, "neutral": 0}

        for token in doc:
            for emotion, keywords in emotion_words.items():
                if token.lemma_ in keywords:
                    emotions[emotion] += 1

        return emotions

    def _aspect_sentiment(self, text: str, aspect: str) -> Optional[float]:
        """Extract sentiment for specific aspects"""
        # Simple aspect-based sentiment (could be enhanced with more sophisticated methods)
        aspect_patterns = {
            "method": ["method", "approach", "technique", "procedure"],
            "result": ["result", "finding", "outcome", "effect"],
            "conclusion": ["conclusion", "summary", "implication"],
            "limitation": ["limitation", "constraint", "drawback"],
            "future work": ["future", "next", "further", "upcoming"]
        }

        patterns = aspect_patterns.get(aspect, [aspect])

        # Find sentences containing aspect keywords
        sentences = text.split('.')
        relevant_sentences = []

        for sentence in sentences:
            if any(pattern in sentence.lower() for pattern in patterns):
                relevant_sentences.append(sentence)

        if relevant_sentences:
            combined_text = '. '.join(relevant_sentences)
            blob = TextBlob(combined_text)
            return blob.sentiment.polarity

        return None

    def _find_optimal_clusters(self, embeddings: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using elbow method"""
        if len(embeddings) < 4:
            return 2

        inertias = []
        k_range = range(2, min(max_k + 1, len(embeddings)))

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)

        # Simple elbow detection
        if len(inertias) < 2:
            return 2

        # Find the point where improvement starts to diminish
        improvements = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]

        if improvements:
            # Find the elbow (where improvement drops significantly)
            avg_improvement = np.mean(improvements)
            for i, improvement in enumerate(improvements):
                if improvement < avg_improvement * 0.5:  # 50% of average improvement
                    return i + 2  # +2 because range starts at 2

        return min(5, max_k)  # Default fallback

    def _find_optimal_eps(self, embeddings: np.ndarray) -> float:
        """Find optimal eps parameter for DBSCAN"""
        from sklearn.neighbors import NearestNeighbors

        neighbors = NearestNeighbors(n_neighbors=2)
        neighbors.fit(embeddings)
        distances, indices = neighbors.kneighbors(embeddings)
        distances = np.sort(distances[:, 1])

        # Use 95th percentile as a reasonable eps value
        return float(np.percentile(distances, 95))

    def _generate_cluster_summaries(self, texts: List[str], labels: List[int]) -> Dict[int, Dict]:
        """Generate summaries for each cluster"""
        cluster_summaries = {}
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:  # Noise cluster in DBSCAN
                continue

            cluster_texts = [texts[i] for i, l in enumerate(labels) if l == label]

            if cluster_texts:
                # Simple keyword extraction for summary
                combined_text = ' '.join(cluster_texts)
                blob = TextBlob(combined_text)

                # Get most common words
                words = [word.lower() for word in blob.words if len(word) > 3]
                common_words = Counter(words).most_common(10)

                cluster_summaries[int(label)] = {
                    "size": len(cluster_texts),
                    "keywords": common_words,
                    "sample_text": cluster_texts[0][:200] + "..." if cluster_texts[0] else ""
                }

        return cluster_summaries

    def _analyze_keyword_evolution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how keywords evolve over time"""
        if 'keywords' not in df.columns:
            return {}

        evolution = {}

        # Get keywords by year
        for year in sorted(df['year'].unique()):
            year_data = df[df['year'] == year]
            year_keywords = []

            for keywords_list in year_data['keywords'].dropna():
                if isinstance(keywords_list, list):
                    year_keywords.extend(keywords_list)
                elif isinstance(keywords_list, str):
                    year_keywords.extend(keywords_list.split(','))

            if year_keywords:
                evolution[int(year)] = Counter(year_keywords).most_common(10)

        return evolution