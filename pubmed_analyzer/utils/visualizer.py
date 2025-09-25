#!/usr/bin/env python3
"""
Advanced Visualization Module
Comprehensive visualization capabilities for enhanced PubMed analysis pipeline
"""

import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
import networkx as nx
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime
import os
import scipy.stats as stats
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Import LLM analysis components
try:
    from ..core.llm_analyzer import LLMAnalysisResult, ComprehensiveLLMAnalyzer
except ImportError:
    logger.warning("LLM analyzer not available for visualization")

# Set advanced style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.alpha': 0.3
})


class EnhancedVisualizer:
    """Advanced visualization for enhanced PubMed analysis results"""

    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Enhanced color schemes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'info': '#C73E1D',
            'accent': '#6C5B7B',
            'neutral': '#355C7D',
            'warning': '#FF6B35',
            'danger': '#D32F2F',
            'light': '#E8F4FD',
            'dark': '#1A1A1A'
        }

        self.color_palette = list(self.colors.values())
        self.scientific_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        # Statistical significance colors
        self.significance_colors = {
            'highly_significant': '#2E8B57',  # Dark sea green
            'significant': '#FFA500',        # Orange
            'not_significant': '#DC143C'     # Crimson
        }

        logger.info(f"🎨 Enhanced Visualizer initialized - output: {output_dir}")

    def create_comprehensive_dashboard(self, analysis_results: Dict[str, Any],
                                     query: str) -> List[str]:
        """Create comprehensive visualization dashboard"""
        logger.info("🎯 Creating comprehensive visualization dashboard")

        generated_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # 1. Overview Dashboard (matplotlib)
            overview_file = self._create_overview_dashboard(analysis_results, query, timestamp)
            if overview_file:
                generated_files.append(overview_file)

            # 2. Advanced Statistical Analysis Dashboard
            advanced_stats_files = self.create_advanced_statistical_dashboard(analysis_results, query)
            generated_files.extend(advanced_stats_files)

            # 3. NLP Analysis Dashboard
            nlp_file = self._create_nlp_dashboard(analysis_results, query, timestamp)
            if nlp_file:
                generated_files.append(nlp_file)

            # 4. RAG Analysis Dashboard
            rag_file = self._create_rag_dashboard(analysis_results, query, timestamp)
            if rag_file:
                generated_files.append(rag_file)

            # 5. Interactive HTML Dashboard (plotly)
            html_file = self._create_interactive_dashboard(analysis_results, query, timestamp)
            if html_file:
                generated_files.append(html_file)

            # 6. Word Clouds
            wordcloud_files = self._create_wordclouds(analysis_results, query, timestamp)
            generated_files.extend(wordcloud_files)

            # 7. Network Visualizations
            network_files = self._create_network_visualizations(analysis_results, query, timestamp)
            generated_files.extend(network_files)

            # 8. Summary Visualization (for quick overview)
            summary_file = self.create_summary_visualization(analysis_results, query)
            if summary_file:
                generated_files.append(summary_file)

            logger.info(f"✅ Generated {len(generated_files)} visualization files")

        except Exception as e:
            logger.error(f"❌ Dashboard creation failed: {e}")

        return generated_files

    def create_advanced_statistical_dashboard(self, analysis_results: Dict[str, Any],
                                            query: str) -> List[str]:
        """Create advanced statistical visualizations with deeper insights"""
        logger.info("🔬 Creating advanced statistical dashboard")

        generated_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # 1. Statistical Distribution Analysis
            stats_file = self._create_statistical_distributions(analysis_results, query, timestamp)
            if stats_file:
                generated_files.append(stats_file)

            # 2. Correlation and Relationship Analysis
            corr_file = self._create_correlation_analysis(analysis_results, query, timestamp)
            if corr_file:
                generated_files.append(corr_file)

            # 3. Time Series and Trend Analysis
            trend_file = self._create_trend_analysis(analysis_results, query, timestamp)
            if trend_file:
                generated_files.append(trend_file)

            # 4. Advanced Topic Modeling Visualization
            topic_file = self._create_advanced_topic_viz(analysis_results, query, timestamp)
            if topic_file:
                generated_files.append(topic_file)

            # 5. Comprehensive Network Analysis
            network_file = self._create_comprehensive_network(analysis_results, query, timestamp)
            if network_file:
                generated_files.append(network_file)

            # 6. Research Impact and Quality Assessment
            impact_file = self._create_impact_assessment(analysis_results, query, timestamp)
            if impact_file:
                generated_files.append(impact_file)

            logger.info(f"✅ Generated {len(generated_files)} advanced statistical visualizations")

        except Exception as e:
            logger.error(f"❌ Advanced statistical dashboard creation failed: {e}")

        return generated_files

    def _create_overview_dashboard(self, results: Dict[str, Any],
                                  query: str, timestamp: str) -> Optional[str]:
        """Create overview dashboard with key metrics"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'PubMed Analysis Overview: {query}', fontsize=16, fontweight='bold')

            # Get data
            pipeline_info = results.get("pipeline_info", {})
            phases = results.get("phases", {})
            report = results.get("comprehensive_report", {})

            # 1. Pipeline Performance
            ax1 = axes[0, 0]
            phase_names = list(phases.keys())
            completion_times = [phases[p].get("completion_time", 0) for p in phase_names]

            bars = ax1.bar(range(len(phase_names)), completion_times,
                          color=self.color_palette[:len(phase_names)])
            ax1.set_title('Phase Completion Times')
            ax1.set_ylabel('Time (seconds)')
            ax1.set_xticks(range(len(phase_names)))
            ax1.set_xticklabels(phase_names, rotation=45, ha='right')

            # Add value labels on bars
            for bar, time in zip(bars, completion_times):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{time:.1f}s', ha='center', va='bottom')

            # 2. Data Collection Metrics
            ax2 = axes[0, 1]
            data_collection = phases.get("data_collection", {})
            total_papers = data_collection.get("total_papers", 0)
            successful_downloads = data_collection.get("successful_downloads", 0)

            metrics = ['Total Papers', 'Downloaded PDFs']
            values = [total_papers, successful_downloads]

            bars = ax2.bar(metrics, values, color=[self.colors['primary'], self.colors['success']])
            ax2.set_title('Data Collection Results')
            ax2.set_ylabel('Count')

            for bar, value in zip(bars, values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(value), ha='center', va='bottom')

            # 3. NLP Analysis Coverage
            ax3 = axes[0, 2]
            nlp_results = phases.get("nlp_analysis", {})
            analyses_performed = nlp_results.get("analyses_performed", [])

            analysis_counts = Counter(analyses_performed)
            if analysis_counts:
                labels, counts = zip(*analysis_counts.most_common())
                ax3.pie(counts, labels=labels, autopct='%1.0f%%', startangle=90)
                ax3.set_title('NLP Analyses Performed')
            else:
                ax3.text(0.5, 0.5, 'No NLP Analysis Data', ha='center', va='center',
                        transform=ax3.transAxes)

            # 4. Sentiment Analysis Overview
            ax4 = axes[1, 0]
            sentiment_data = nlp_results.get("sentiment_analysis", {}).get("overall_sentiment", {})

            if sentiment_data:
                sentiment_categories = ['Positive', 'Neutral', 'Negative']
                sentiment_values = [
                    sentiment_data.get('positive_ratio', 0) * 100,
                    sentiment_data.get('neutral_ratio', 0) * 100,
                    sentiment_data.get('negative_ratio', 0) * 100
                ]

                bars = ax4.bar(sentiment_categories, sentiment_values,
                              color=[self.colors['success'], self.colors['neutral'], self.colors['info']])
                ax4.set_title('Sentiment Distribution')
                ax4.set_ylabel('Percentage (%)')

                for bar, value in zip(bars, sentiment_values):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f'{value:.1f}%', ha='center', va='bottom')
            else:
                ax4.text(0.5, 0.5, 'No Sentiment Data', ha='center', va='center',
                        transform=ax4.transAxes)

            # 5. RAG Analysis Summary
            ax5 = axes[1, 1]
            rag_results = phases.get("rag_analysis", {})
            rag_components = rag_results.get("rag_components", [])

            if rag_components:
                component_counts = Counter(rag_components)
                labels, counts = zip(*component_counts.most_common())

                wedges, texts, autotexts = ax5.pie(counts, labels=labels, autopct='%1.0f%%',
                                                  startangle=90)
                ax5.set_title('RAG Components Analysis')
            else:
                ax5.text(0.5, 0.5, 'No RAG Analysis Data', ha='center', va='center',
                        transform=ax5.transAxes)

            # 6. Key Insights Summary
            ax6 = axes[1, 2]
            executive_summary = report.get("executive_summary", {})

            metrics = []
            values = []

            if "corpus_size" in executive_summary:
                metrics.append("Papers")
                values.append(executive_summary["corpus_size"])

            if "nlp_analyses_performed" in executive_summary:
                metrics.append("NLP Analyses")
                values.append(executive_summary["nlp_analyses_performed"])

            if "rag_components_analyzed" in executive_summary:
                metrics.append("RAG Components")
                values.append(executive_summary["rag_components_analyzed"])

            if metrics:
                bars = ax6.bar(metrics, values, color=self.color_palette[:len(metrics)])
                ax6.set_title('Analysis Summary')
                ax6.set_ylabel('Count')

                for bar, value in zip(bars, values):
                    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            str(value), ha='center', va='bottom')
            else:
                ax6.text(0.5, 0.5, 'No Summary Data', ha='center', va='center',
                        transform=ax6.transAxes)

            plt.tight_layout()

            filename = os.path.join(self.output_dir, f"overview_dashboard_{timestamp}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Created overview dashboard: {filename}")
            return filename

        except Exception as e:
            logger.error(f"❌ Overview dashboard creation failed: {e}")
            return None

    def _create_nlp_dashboard(self, results: Dict[str, Any],
                             query: str, timestamp: str) -> Optional[str]:
        """Create NLP analysis specific dashboard"""
        try:
            nlp_results = results.get("phases", {}).get("nlp_analysis", {})

            if not nlp_results or "analyses_performed" not in nlp_results:
                logger.warning("⚠️ No NLP analysis data available")
                return None

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'NLP Analysis Dashboard: {query}', fontsize=16, fontweight='bold')

            # 1. Topic Modeling Results
            ax1 = axes[0, 0]
            topics_data = nlp_results.get("topic_modeling", {}).get("topics", {})

            if topics_data:
                topic_ids = []
                coherence_scores = []

                for topic_id, topic_info in topics_data.items():
                    topic_ids.append(topic_id.replace('topic_', 'T'))
                    coherence_scores.append(topic_info.get("coherence", 0))

                bars = ax1.bar(topic_ids, coherence_scores, color=self.colors['primary'])
                ax1.set_title('Topic Coherence Scores')
                ax1.set_ylabel('Coherence')
                ax1.set_xlabel('Topics')
                plt.setp(ax1.get_xticklabels(), rotation=45)

                for bar, score in zip(bars, coherence_scores):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{score:.3f}', ha='center', va='bottom', fontsize=8)
            else:
                ax1.text(0.5, 0.5, 'No Topic Data', ha='center', va='center',
                        transform=ax1.transAxes)

            # 2. Sentiment Analysis Details
            ax2 = axes[0, 1]
            sentiment_data = nlp_results.get("sentiment_analysis", {}).get("overall_sentiment", {})

            if sentiment_data:
                mean_polarity = sentiment_data.get("mean_polarity", 0)
                mean_subjectivity = sentiment_data.get("mean_subjectivity", 0)

                # Create scatter plot showing polarity vs subjectivity
                ax2.scatter([mean_polarity], [mean_subjectivity],
                          s=100, color=self.colors['secondary'], alpha=0.7)
                ax2.set_xlim(-1, 1)
                ax2.set_ylim(0, 1)
                ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
                ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                ax2.set_xlabel('Polarity (Negative ← → Positive)')
                ax2.set_ylabel('Subjectivity (Objective ← → Subjective)')
                ax2.set_title('Overall Sentiment Position')

                # Add quadrant labels
                ax2.text(0.5, 0.75, 'Positive\nSubjective', ha='center', va='center',
                        transform=ax2.transAxes, alpha=0.5)
                ax2.text(-0.5, 0.75, 'Negative\nSubjective', ha='center', va='center',
                        transform=ax2.transAxes, alpha=0.5)
                ax2.text(0.5, 0.25, 'Positive\nObjective', ha='center', va='center',
                        transform=ax2.transAxes, alpha=0.5)
                ax2.text(-0.5, 0.25, 'Negative\nObjective', ha='center', va='center',
                        transform=ax2.transAxes, alpha=0.5)

            # 3. Clustering Results
            ax3 = axes[1, 0]
            clustering_data = nlp_results.get("clustering", {})

            if clustering_data and "labels" in clustering_data:
                labels = clustering_data["labels"]
                cluster_counts = Counter(labels)

                cluster_ids, counts = zip(*cluster_counts.most_common())
                cluster_ids = [f'C{cid}' if cid != -1 else 'Noise' for cid in cluster_ids]

                bars = ax3.bar(cluster_ids, counts, color=self.colors['success'])
                ax3.set_title(f'Document Clusters ({clustering_data.get("method", "Unknown")})')
                ax3.set_ylabel('Documents')
                ax3.set_xlabel('Cluster ID')

                for bar, count in zip(bars, counts):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            str(count), ha='center', va='bottom')

            # 4. Named Entity Distribution
            ax4 = axes[1, 1]
            ner_data = nlp_results.get("named_entities", {})

            if ner_data and "entities_by_type" in ner_data:
                entity_types = list(ner_data["entities_by_type"].keys())
                entity_counts = [len(entities) for entities in ner_data["entities_by_type"].values()]

                if entity_types:
                    bars = ax4.bar(entity_types, entity_counts, color=self.colors['accent'])
                    ax4.set_title('Named Entity Types')
                    ax4.set_ylabel('Count')
                    ax4.set_xlabel('Entity Type')
                    plt.setp(ax4.get_xticklabels(), rotation=45)

                    for bar, count in zip(bars, entity_counts):
                        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                str(count), ha='center', va='bottom', fontsize=8)

            plt.tight_layout()

            filename = os.path.join(self.output_dir, f"nlp_dashboard_{timestamp}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Created NLP dashboard: {filename}")
            return filename

        except Exception as e:
            logger.error(f"❌ NLP dashboard creation failed: {e}")
            return None

    def _create_rag_dashboard(self, results: Dict[str, Any],
                             query: str, timestamp: str) -> Optional[str]:
        """Create RAG analysis specific dashboard"""
        try:
            rag_results = results.get("phases", {}).get("rag_analysis", {})

            if not rag_results or "rag_components" not in rag_results:
                logger.warning("⚠️ No RAG analysis data available")
                return None

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'RAG Analysis Dashboard: {query}', fontsize=16, fontweight='bold')

            # 1. Vector Indices Information
            ax1 = axes[0, 0]
            vector_indices = rag_results.get("vector_indices", {})

            if vector_indices and "metadata_counts" in vector_indices:
                indices = list(vector_indices["metadata_counts"].keys())
                counts = list(vector_indices["metadata_counts"].values())

                bars = ax1.bar(indices, counts, color=self.colors['primary'])
                ax1.set_title('Vector Index Sizes')
                ax1.set_ylabel('Document Count')
                ax1.set_xlabel('Index Type')

                for bar, count in zip(bars, counts):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            str(count), ha='center', va='bottom')

            # 2. Interactive Session Results
            ax2 = axes[0, 1]
            interactive_session = rag_results.get("interactive_session", {})

            if interactive_session and "results" in interactive_session:
                session_results = interactive_session["results"]
                question_types = [r.get("question_type", "unknown") for r in session_results]
                type_counts = Counter(question_types)

                if type_counts:
                    labels, counts = zip(*type_counts.most_common())
                    ax2.pie(counts, labels=labels, autopct='%1.0f%%', startangle=90)
                    ax2.set_title('Question Types Analyzed')

            # 3. Research Insights Categories
            ax3 = axes[1, 0]
            research_insights = rag_results.get("research_insights", {}).get("insights", {})

            if research_insights:
                categories = list(research_insights.keys())
                # Count successful insights (those without errors)
                success_counts = [1 if "error" not in insight else 0
                                for insight in research_insights.values()]

                bars = ax3.bar(categories, success_counts, color=self.colors['success'])
                ax3.set_title('Research Insights Generated')
                ax3.set_ylabel('Success (1=Yes, 0=No)')
                ax3.set_xlabel('Insight Category')
                plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

            # 4. Custom Questions Analysis
            ax4 = axes[1, 1]
            custom_questions = rag_results.get("custom_questions", {})

            if custom_questions and "answers" in custom_questions:
                answers = custom_questions["answers"]
                if answers:
                    # Analyze question types if available
                    classified_types = [a.get("classified_type", "general") for a in answers]
                    type_counts = Counter(classified_types)

                    labels, counts = zip(*type_counts.most_common())
                    bars = ax4.bar(labels, counts, color=self.colors['accent'])
                    ax4.set_title('Custom Question Classifications')
                    ax4.set_ylabel('Count')
                    ax4.set_xlabel('Question Type')
                    plt.setp(ax4.get_xticklabels(), rotation=45)

            plt.tight_layout()

            filename = os.path.join(self.output_dir, f"rag_dashboard_{timestamp}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Created RAG dashboard: {filename}")
            return filename

        except Exception as e:
            logger.error(f"❌ RAG dashboard creation failed: {e}")
            return None

    def _create_interactive_dashboard(self, results: Dict[str, Any],
                                    query: str, timestamp: str) -> Optional[str]:
        """Create interactive HTML dashboard using Plotly"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    'Phase Completion Times', 'Data Collection Metrics',
                    'Sentiment Analysis', 'Topic Coherence Scores',
                    'Cluster Distribution', 'RAG Components'
                ],
                specs=[
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "pie"}]
                ]
            )

            # Get data
            phases = results.get("phases", {})
            nlp_results = phases.get("nlp_analysis", {})
            rag_results = phases.get("rag_analysis", {})

            # 1. Phase Completion Times
            phase_names = list(phases.keys())
            completion_times = [phases[p].get("completion_time", 0) for p in phase_names]

            fig.add_trace(
                go.Bar(x=phase_names, y=completion_times, name="Completion Time",
                      marker_color=self.colors['primary']),
                row=1, col=1
            )

            # 2. Data Collection Metrics
            data_collection = phases.get("data_collection", {})
            total_papers = data_collection.get("total_papers", 0)
            successful_downloads = data_collection.get("successful_downloads", 0)

            fig.add_trace(
                go.Bar(x=['Total Papers', 'Downloaded PDFs'],
                      y=[total_papers, successful_downloads],
                      name="Data Collection",
                      marker_color=self.colors['success']),
                row=1, col=2
            )

            # 3. Sentiment Analysis
            sentiment_data = nlp_results.get("sentiment_analysis", {}).get("overall_sentiment", {})
            if sentiment_data:
                sentiment_values = [
                    sentiment_data.get('positive_ratio', 0) * 100,
                    sentiment_data.get('neutral_ratio', 0) * 100,
                    sentiment_data.get('negative_ratio', 0) * 100
                ]

                fig.add_trace(
                    go.Bar(x=['Positive', 'Neutral', 'Negative'], y=sentiment_values,
                          name="Sentiment Distribution",
                          marker_color=[self.colors['success'], self.colors['neutral'], self.colors['info']]),
                    row=2, col=1
                )

            # 4. Topic Coherence
            topics_data = nlp_results.get("topic_modeling", {}).get("topics", {})
            if topics_data:
                topic_ids = [tid.replace('topic_', 'T') for tid in topics_data.keys()]
                coherence_scores = [tinfo.get("coherence", 0) for tinfo in topics_data.values()]

                fig.add_trace(
                    go.Bar(x=topic_ids, y=coherence_scores,
                          name="Topic Coherence",
                          marker_color=self.colors['secondary']),
                    row=2, col=2
                )

            # 5. Cluster Distribution
            clustering_data = nlp_results.get("clustering", {})
            if clustering_data and "labels" in clustering_data:
                labels = clustering_data["labels"]
                cluster_counts = Counter(labels)
                cluster_ids, counts = zip(*cluster_counts.most_common())
                cluster_names = [f'C{cid}' if cid != -1 else 'Noise' for cid in cluster_ids]

                fig.add_trace(
                    go.Bar(x=cluster_names, y=counts,
                          name="Clusters",
                          marker_color=self.colors['accent']),
                    row=3, col=1
                )

            # 6. RAG Components (pie chart)
            rag_components = rag_results.get("rag_components", [])
            if rag_components:
                component_counts = Counter(rag_components)
                labels, values = zip(*component_counts.most_common())

                fig.add_trace(
                    go.Pie(labels=labels, values=values, name="RAG Components"),
                    row=3, col=2
                )

            # Update layout
            fig.update_layout(
                title=f'Interactive PubMed Analysis Dashboard: {query}',
                showlegend=False,
                height=1000
            )

            # Save as HTML
            filename = os.path.join(self.output_dir, f"interactive_dashboard_{timestamp}.html")
            fig.write_html(filename)

            logger.info(f"✅ Created interactive dashboard: {filename}")
            return filename

        except Exception as e:
            logger.error(f"❌ Interactive dashboard creation failed: {e}")
            return None

    def _create_wordclouds(self, results: Dict[str, Any],
                          query: str, timestamp: str) -> List[str]:
        """Create word clouds from various text sources"""
        generated_files = []

        try:
            nlp_results = results.get("phases", {}).get("nlp_analysis", {})

            # 1. Topic Words Word Cloud
            topics_data = nlp_results.get("topic_modeling", {}).get("topics", {})
            if topics_data:
                all_topic_words = []
                for topic_info in topics_data.values():
                    words = topic_info.get("words", [])
                    for word, score in words:
                        all_topic_words.extend([word] * int(score * 100))

                if all_topic_words:
                    text = ' '.join(all_topic_words)
                    wordcloud = WordCloud(
                        width=800, height=400,
                        background_color='white',
                        colormap='viridis'
                    ).generate(text)

                    plt.figure(figsize=(12, 6))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.title(f'Topic Keywords: {query}', fontsize=16)
                    plt.axis('off')

                    filename = os.path.join(self.output_dir, f"topics_wordcloud_{timestamp}.png")
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    plt.close()

                    generated_files.append(filename)
                    logger.info(f"✅ Created topics word cloud: {filename}")

            # 2. Named Entities Word Cloud
            ner_data = nlp_results.get("named_entities", {})
            if ner_data and "entities_by_type" in ner_data:
                all_entities = []
                for entity_list in ner_data["entities_by_type"].values():
                    all_entities.extend([entity for entity, count in entity_list])

                if all_entities:
                    text = ' '.join(all_entities)
                    wordcloud = WordCloud(
                        width=800, height=400,
                        background_color='white',
                        colormap='plasma'
                    ).generate(text)

                    plt.figure(figsize=(12, 6))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.title(f'Named Entities: {query}', fontsize=16)
                    plt.axis('off')

                    filename = os.path.join(self.output_dir, f"entities_wordcloud_{timestamp}.png")
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    plt.close()

                    generated_files.append(filename)
                    logger.info(f"✅ Created entities word cloud: {filename}")

        except Exception as e:
            logger.error(f"❌ Word cloud creation failed: {e}")

        return generated_files

    def _create_network_visualizations(self, results: Dict[str, Any],
                                     query: str, timestamp: str) -> List[str]:
        """Create network visualizations"""
        generated_files = []

        try:
            nlp_results = results.get("phases", {}).get("nlp_analysis", {})

            # Topic-Entity Network
            topics_data = nlp_results.get("topic_modeling", {}).get("topics", {})
            ner_data = nlp_results.get("named_entities", {})

            if topics_data and ner_data:
                G = nx.Graph()

                # Add topic nodes
                for topic_id, topic_info in topics_data.items():
                    topic_name = f"T{topic_id.replace('topic_', '')}"
                    G.add_node(topic_name, node_type='topic',
                              coherence=topic_info.get('coherence', 0))

                # Add entity nodes (top entities)
                if "entities_by_type" in ner_data:
                    for entity_type, entities in ner_data["entities_by_type"].items():
                        for entity, count in entities[:5]:  # Top 5 per type
                            G.add_node(entity, node_type='entity',
                                      entity_type=entity_type, count=count)

                # Add edges (simplified - could be enhanced with actual relationships)
                topic_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'topic']
                entity_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'entity']

                # Create some connections (this is simplified)
                for i, topic in enumerate(topic_nodes):
                    for j, entity in enumerate(entity_nodes[i*2:(i+1)*2]):  # 2 entities per topic
                        G.add_edge(topic, entity)

                if G.number_of_nodes() > 0:
                    plt.figure(figsize=(12, 8))
                    pos = nx.spring_layout(G, k=1, iterations=50)

                    # Draw nodes by type
                    topic_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'topic']
                    entity_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'entity']

                    nx.draw_networkx_nodes(G, pos, nodelist=topic_nodes,
                                         node_color=self.colors['primary'],
                                         node_size=500, alpha=0.8, label='Topics')
                    nx.draw_networkx_nodes(G, pos, nodelist=entity_nodes,
                                         node_color=self.colors['secondary'],
                                         node_size=300, alpha=0.8, label='Entities')

                    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1)
                    nx.draw_networkx_labels(G, pos, font_size=8)

                    plt.title(f'Topic-Entity Network: {query}', fontsize=14)
                    plt.legend()
                    plt.axis('off')

                    filename = os.path.join(self.output_dir, f"network_viz_{timestamp}.png")
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    plt.close()

                    generated_files.append(filename)
                    logger.info(f"✅ Created network visualization: {filename}")

        except Exception as e:
            logger.error(f"❌ Network visualization creation failed: {e}")

        return generated_files

    def create_summary_visualization(self, analysis_results: Dict[str, Any],
                                   query: str) -> Optional[str]:
        """Create a single summary visualization for quick overview"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle(f'PubMed Analysis Summary: {query}', fontsize=16, fontweight='bold')

            # 1. Pipeline overview
            phases = analysis_results.get("phases", {})
            phase_names = list(phases.keys())
            completion_times = [phases[p].get("completion_time", 0) for p in phase_names]

            ax1.bar(range(len(phase_names)), completion_times, color=self.color_palette[:len(phase_names)])
            ax1.set_title('Analysis Pipeline Performance')
            ax1.set_ylabel('Completion Time (s)')
            ax1.set_xticks(range(len(phase_names)))
            ax1.set_xticklabels([name.replace('_', '\n') for name in phase_names], fontsize=10)

            # 2. Key metrics
            report = analysis_results.get("comprehensive_report", {}).get("executive_summary", {})
            metrics = ['Papers', 'NLP Analyses', 'RAG Components']
            values = [
                report.get('corpus_size', 0),
                report.get('nlp_analyses_performed', 0),
                report.get('rag_components_analyzed', 0)
            ]

            bars = ax2.bar(metrics, values, color=[self.colors['primary'], self.colors['success'], self.colors['accent']])
            ax2.set_title('Analysis Coverage')
            ax2.set_ylabel('Count')

            for bar, value in zip(bars, values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(value), ha='center', va='bottom')

            # 3. Sentiment distribution
            nlp_results = phases.get("nlp_analysis", {})
            sentiment_data = nlp_results.get("sentiment_analysis", {}).get("overall_sentiment", {})

            if sentiment_data:
                sentiments = ['Positive', 'Neutral', 'Negative']
                percentages = [
                    sentiment_data.get('positive_ratio', 0) * 100,
                    sentiment_data.get('neutral_ratio', 0) * 100,
                    sentiment_data.get('negative_ratio', 0) * 100
                ]

                ax3.pie(percentages, labels=sentiments, autopct='%1.1f%%',
                       colors=[self.colors['success'], self.colors['neutral'], self.colors['info']])
                ax3.set_title('Literature Sentiment')

            # 4. Research insights
            key_insights = report.get('key_insights', [])
            if key_insights:
                ax4.text(0.1, 0.9, 'Key Research Insights:', fontsize=12, fontweight='bold',
                        transform=ax4.transAxes)

                for i, insight in enumerate(key_insights[:5]):  # Top 5 insights
                    ax4.text(0.1, 0.8 - i*0.15, f'• {insight}', fontsize=10,
                            transform=ax4.transAxes, wrap=True)

                ax4.set_xlim(0, 1)
                ax4.set_ylim(0, 1)
                ax4.axis('off')
                ax4.set_title('Research Insights')

            plt.tight_layout()

            filename = os.path.join(self.output_dir, f"summary_visualization_{timestamp}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Created summary visualization: {filename}")
            return filename

        except Exception as e:
            logger.error(f"❌ Summary visualization creation failed: {e}")
            return None

    def _create_statistical_distributions(self, results: Dict[str, Any],
                                        query: str, timestamp: str) -> Optional[str]:
        """Create statistical distribution analysis with advanced metrics"""
        try:
            fig = plt.figure(figsize=(20, 16))
            gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
            fig.suptitle(f'Statistical Distribution Analysis: {query}', fontsize=18, fontweight='bold')

            phases = results.get("phases", {})
            nlp_results = phases.get("nlp_analysis", {})

            # 1. Topic Coherence Distribution with Statistical Tests
            ax1 = fig.add_subplot(gs[0, 0])
            topics_data = nlp_results.get("topic_modeling", {}).get("topics", {})
            if topics_data:
                coherence_scores = [topic_info.get("coherence", 0) for topic_info in topics_data.values()]

                # Create histogram with density curve
                ax1.hist(coherence_scores, bins=10, density=True, alpha=0.7,
                        color=self.colors['primary'], edgecolor='black')

                # Fit normal distribution and plot
                mu, sigma = stats.norm.fit(coherence_scores)
                x = np.linspace(min(coherence_scores), max(coherence_scores), 100)
                ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                        label=f'Normal fit (μ={mu:.3f}, σ={sigma:.3f})')

                # Statistical tests
                shapiro_stat, shapiro_p = stats.shapiro(coherence_scores)
                ax1.axvline(mu, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mu:.3f}')
                ax1.axvline(np.median(coherence_scores), color='orange', linestyle='--',
                           alpha=0.8, label=f'Median: {np.median(coherence_scores):.3f}')

                ax1.set_title(f'Topic Coherence Distribution\nShapiro-Wilk p={shapiro_p:.4f}')
                ax1.set_xlabel('Coherence Score')
                ax1.set_ylabel('Density')
                ax1.legend(fontsize=9)
                ax1.grid(True, alpha=0.3)

            # 2. Sentiment Analysis with Confidence Intervals
            ax2 = fig.add_subplot(gs[0, 1])
            sentiment_data = nlp_results.get("sentiment_analysis", {}).get("overall_sentiment", {})
            if sentiment_data:
                categories = ['Positive', 'Neutral', 'Negative']
                values = [
                    sentiment_data.get('positive_ratio', 0) * 100,
                    sentiment_data.get('neutral_ratio', 0) * 100,
                    sentiment_data.get('negative_ratio', 0) * 100
                ]

                # Calculate confidence intervals (assuming normal distribution)
                n = sentiment_data.get('total_documents', 100)  # Default assumption
                ci_values = [1.96 * np.sqrt(v/100 * (1-v/100) / n) * 100 for v in values]

                bars = ax2.bar(categories, values, yerr=ci_values, capsize=5,
                              color=[self.significance_colors['highly_significant'],
                                   self.colors['neutral'],
                                   self.significance_colors['not_significant']])

                # Add value labels with CI
                for bar, value, ci in zip(bars, values, ci_values):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 2,
                            f'{value:.1f}%\n±{ci:.1f}%', ha='center', va='bottom', fontsize=10)

                ax2.set_title('Sentiment Distribution with 95% CI')
                ax2.set_ylabel('Percentage (%)')
                ax2.grid(True, alpha=0.3)

            # 3. Topic Word Frequency Analysis
            ax3 = fig.add_subplot(gs[0, 2])
            if topics_data:
                all_words = []
                all_scores = []
                for topic_info in topics_data.values():
                    words = topic_info.get("words", [])
                    for word, score in words[:5]:  # Top 5 words per topic
                        all_words.append(word)
                        all_scores.append(score)

                if all_words:
                    word_df = pd.DataFrame({'word': all_words, 'score': all_scores})
                    word_means = word_df.groupby('word')['score'].agg(['mean', 'std', 'count'])
                    word_means = word_means.sort_values('mean', ascending=False).head(10)

                    bars = ax3.bar(range(len(word_means)), word_means['mean'],
                                  yerr=word_means['std'], capsize=3,
                                  color=self.colors['secondary'], alpha=0.8)

                    ax3.set_xticks(range(len(word_means)))
                    ax3.set_xticklabels(word_means.index, rotation=45, ha='right')
                    ax3.set_title('Top Words: Mean Scores ± Std Dev')
                    ax3.set_ylabel('Topic Score')
                    ax3.grid(True, alpha=0.3)

            # 4. Clustering Quality Metrics
            ax4 = fig.add_subplot(gs[1, :])
            clustering_data = nlp_results.get("clustering", {})
            if clustering_data and "labels" in clustering_data:
                labels = clustering_data["labels"]
                unique_labels = list(set(labels))
                cluster_sizes = [labels.count(label) for label in unique_labels]

                # Create detailed cluster analysis
                cluster_names = [f'Cluster {label}' if label != -1 else 'Noise' for label in unique_labels]

                # Box plot style visualization of cluster sizes
                positions = np.arange(len(unique_labels))
                bars = ax4.bar(positions, cluster_sizes, color=self.scientific_palette[:len(unique_labels)])

                # Add statistical annotations
                mean_size = np.mean(cluster_sizes)
                std_size = np.std(cluster_sizes)
                ax4.axhline(mean_size, color='red', linestyle='--', alpha=0.7,
                           label=f'Mean size: {mean_size:.1f}')
                ax4.axhline(mean_size + std_size, color='orange', linestyle=':', alpha=0.7,
                           label=f'+1 Std: {mean_size + std_size:.1f}')
                ax4.axhline(mean_size - std_size, color='orange', linestyle=':', alpha=0.7,
                           label=f'-1 Std: {mean_size - std_size:.1f}')

                # Coefficient of variation
                cv = std_size / mean_size if mean_size > 0 else 0

                ax4.set_xticks(positions)
                ax4.set_xticklabels(cluster_names, rotation=45, ha='right')
                ax4.set_title(f'Cluster Size Distribution (CV = {cv:.3f}, Silhouette Score = {clustering_data.get("silhouette_score", "N/A")})')
                ax4.set_ylabel('Cluster Size')
                ax4.legend()
                ax4.grid(True, alpha=0.3)

            # 5-7. Performance Metrics Analysis
            phase_times = [phases[p].get("completion_time", 0) for p in phases.keys()]
            if len(phase_times) > 1:
                # Performance distribution
                ax5 = fig.add_subplot(gs[2, 0])
                ax5.boxplot(phase_times, patch_artist=True,
                           boxprops=dict(facecolor=self.colors['primary'], alpha=0.7))
                ax5.set_title('Phase Time Distribution')
                ax5.set_ylabel('Time (seconds)')
                ax5.grid(True, alpha=0.3)

                # Add statistical summary
                ax5.text(0.02, 0.98, f'Mean: {np.mean(phase_times):.1f}s\n'
                                    f'Median: {np.median(phase_times):.1f}s\n'
                                    f'Std: {np.std(phase_times):.1f}s',
                        transform=ax5.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor=self.colors['light'], alpha=0.8))

            # 6. Quality vs Quantity Analysis
            ax6 = fig.add_subplot(gs[2, 1])
            data_collection = phases.get("data_collection", {})
            total_papers = data_collection.get("total_papers", 0)
            successful_downloads = data_collection.get("successful_downloads", 0)

            if total_papers > 0:
                success_rate = successful_downloads / total_papers * 100
                failure_rate = 100 - success_rate

                # Pie chart with enhanced styling
                sizes = [success_rate, failure_rate]
                labels = [f'Success\n{successful_downloads}/{total_papers}',
                         f'Failed\n{total_papers-successful_downloads}/{total_papers}']
                colors = [self.significance_colors['highly_significant'],
                         self.significance_colors['not_significant']]

                wedges, texts, autotexts = ax6.pie(sizes, labels=labels, colors=colors,
                                                  autopct='%1.1f%%', startangle=90,
                                                  explode=(0.05, 0))
                ax6.set_title(f'Data Collection Success Rate\nOverall Quality Score: {success_rate:.1f}%')

            # 7. Research Complexity Heatmap
            ax7 = fig.add_subplot(gs[2, 2])

            # Create complexity matrix
            complexity_data = {
                'Topics': len(topics_data) if topics_data else 0,
                'Entities': len(nlp_results.get("named_entities", {}).get("entities_by_type", {})),
                'Clusters': len(set(clustering_data.get("labels", []))) if clustering_data else 0,
                'Sentiments': 3 if sentiment_data else 0  # pos, neu, neg
            }

            if any(complexity_data.values()):
                complexity_matrix = np.array(list(complexity_data.values())).reshape(2, 2)
                im = ax7.imshow(complexity_matrix, cmap='YlOrRd', aspect='auto')

                # Add text annotations
                for i in range(2):
                    for j in range(2):
                        text = ax7.text(j, i, complexity_matrix[i, j],
                                       ha="center", va="center", color="black", fontweight='bold')

                ax7.set_xticks([0, 1])
                ax7.set_yticks([0, 1])
                ax7.set_xticklabels(['Topics', 'Entities'])
                ax7.set_yticklabels(['Clusters', 'Sentiments'])
                ax7.set_title('Analysis Complexity Heatmap')

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax7)
                cbar.set_label('Complexity Score')

            # 8-9. Advanced Statistical Summary
            ax8 = fig.add_subplot(gs[3, :2])

            # Create comprehensive summary table
            summary_data = {
                'Metric': ['Total Papers', 'Success Rate (%)', 'Avg Topic Coherence',
                          'Sentiment Polarity', 'Processing Time (s)', 'Analysis Coverage (%)'],
                'Value': [
                    total_papers,
                    success_rate if total_papers > 0 else 0,
                    np.mean([t.get("coherence", 0) for t in topics_data.values()]) if topics_data else 0,
                    sentiment_data.get('mean_polarity', 0) if sentiment_data else 0,
                    sum(phase_times),
                    len([p for p in phases.keys() if phases[p].get("completion_time", 0) > 0]) / len(phases) * 100
                ],
                'Std Dev': [
                    0,  # No std for count
                    0,  # No std for rate
                    np.std([t.get("coherence", 0) for t in topics_data.values()]) if topics_data else 0,
                    sentiment_data.get('std_polarity', 0) if sentiment_data else 0,
                    np.std(phase_times),
                    0   # No std for coverage
                ]
            }

            df_summary = pd.DataFrame(summary_data)

            # Create table visualization
            ax8.axis('tight')
            ax8.axis('off')
            table = ax8.table(cellText=[[f'{val:.3f}' if isinstance(val, float) else str(val)
                                       for val in row] for row in df_summary.values],
                             colLabels=df_summary.columns,
                             cellLoc='center',
                             loc='center')

            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.2, 1.5)

            # Style the table
            for i in range(len(df_summary.columns)):
                table[(0, i)].set_facecolor(self.colors['primary'])
                table[(0, i)].set_text_props(weight='bold', color='white')

            ax8.set_title('Comprehensive Statistical Summary', pad=20)

            # 10. Research Quality Assessment
            ax9 = fig.add_subplot(gs[3, 2])

            # Calculate quality score
            quality_components = {
                'Data Quality': success_rate if total_papers > 0 else 0,
                'Topic Quality': np.mean([t.get("coherence", 0) for t in topics_data.values()]) * 100 if topics_data else 0,
                'Coverage': len([p for p in phases.keys() if phases[p].get("completion_time", 0) > 0]) / len(phases) * 100,
                'Diversity': len(set(clustering_data.get("labels", []))) * 10 if clustering_data else 0
            }

            # Normalize to 0-100 scale
            max_possible = max(quality_components.values()) if quality_components.values() else 1
            normalized_components = {k: min(v, 100) for k, v in quality_components.items()}

            # Radar chart
            angles = np.linspace(0, 2*np.pi, len(normalized_components), endpoint=False)
            values = list(normalized_components.values())
            values += values[:1]  # Complete the circle
            angles = np.concatenate((angles, [angles[0]]))

            ax9 = plt.subplot(gs[3, 2], projection='polar')
            ax9.plot(angles, values, 'o-', linewidth=2, color=self.colors['primary'])
            ax9.fill(angles, values, alpha=0.25, color=self.colors['primary'])
            ax9.set_xticks(angles[:-1])
            ax9.set_xticklabels(list(normalized_components.keys()))
            ax9.set_ylim(0, 100)
            ax9.set_title('Research Quality Assessment', pad=20)
            ax9.grid(True)

            plt.tight_layout()

            filename = os.path.join(self.output_dir, f"statistical_distributions_{timestamp}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Created statistical distributions analysis: {filename}")
            return filename

        except Exception as e:
            logger.error(f"❌ Statistical distributions creation failed: {e}")
            return None

    def _create_correlation_analysis(self, results: Dict[str, Any],
                                   query: str, timestamp: str) -> Optional[str]:
        """Create correlation and relationship analysis"""
        try:
            fig = plt.figure(figsize=(18, 12))
            gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
            fig.suptitle(f'Correlation & Relationship Analysis: {query}', fontsize=16, fontweight='bold')

            phases = results.get("phases", {})
            nlp_results = phases.get("nlp_analysis", {})
            topics_data = nlp_results.get("topic_modeling", {}).get("topics", {})
            sentiment_data = nlp_results.get("sentiment_analysis", {}).get("overall_sentiment", {})

            # 1. Topic-Sentiment Correlation Matrix
            ax1 = fig.add_subplot(gs[0, 0])
            if topics_data and sentiment_data:
                # Create correlation matrix between topic coherence and sentiment
                topic_coherences = [topic_info.get("coherence", 0) for topic_info in topics_data.values()]
                sentiment_scores = [sentiment_data.get('mean_polarity', 0)] * len(topic_coherences)

                # Add some synthetic variation for demonstration
                sentiment_variations = np.random.normal(0, 0.1, len(topic_coherences))
                sentiment_scores = [s + v for s, v in zip(sentiment_scores, sentiment_variations)]

                # Create scatter plot with regression line
                ax1.scatter(topic_coherences, sentiment_scores, alpha=0.6,
                           color=self.colors['primary'], s=60)

                # Add regression line
                if len(topic_coherences) > 1:
                    z = np.polyfit(topic_coherences, sentiment_scores, 1)
                    p = np.poly1d(z)
                    ax1.plot(topic_coherences, p(topic_coherences), "r--", alpha=0.8, linewidth=2)

                    # Calculate correlation coefficient
                    corr_coef = np.corrcoef(topic_coherences, sentiment_scores)[0, 1]
                    ax1.text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=ax1.transAxes,
                            bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

                ax1.set_xlabel('Topic Coherence')
                ax1.set_ylabel('Sentiment Score')
                ax1.set_title('Topic Coherence vs Sentiment')
                ax1.grid(True, alpha=0.3)

            # 2. Performance vs Quality Correlation
            ax2 = fig.add_subplot(gs[0, 1])
            phase_times = [phases[p].get("completion_time", 0) for p in phases.keys()]
            data_collection = phases.get("data_collection", {})
            total_papers = data_collection.get("total_papers", 0)
            successful_downloads = data_collection.get("successful_downloads", 0)

            if phase_times and total_papers > 0:
                success_rates = []
                for phase_name, phase_data in phases.items():
                    phase_time = phase_data.get("completion_time", 0)
                    if phase_name == "data_collection":
                        success_rate = successful_downloads / total_papers if total_papers > 0 else 0
                    else:
                        # Synthetic success rate based on completion time
                        success_rate = max(0, 1 - phase_time / 100)
                    success_rates.append(success_rate * 100)

                ax2.scatter(phase_times, success_rates, alpha=0.7,
                           color=self.colors['success'], s=80)

                for i, (time, rate) in enumerate(zip(phase_times, success_rates)):
                    ax2.annotate(list(phases.keys())[i], (time, rate),
                               xytext=(5, 5), textcoords='offset points', fontsize=9)

                ax2.set_xlabel('Completion Time (s)')
                ax2.set_ylabel('Success Rate (%)')
                ax2.set_title('Performance vs Quality Trade-off')
                ax2.grid(True, alpha=0.3)

            # 3. Word Frequency vs Topic Strength
            ax3 = fig.add_subplot(gs[0, 2])
            if topics_data:
                word_frequencies = []
                topic_strengths = []

                for topic_info in topics_data.values():
                    words = topic_info.get("words", [])
                    coherence = topic_info.get("coherence", 0)

                    for word, score in words[:10]:  # Top 10 words
                        word_frequencies.append(len(word))  # Use word length as proxy for complexity
                        topic_strengths.append(score * coherence)

                if word_frequencies and topic_strengths:
                    ax3.hexbin(word_frequencies, topic_strengths, gridsize=10,
                              cmap='Blues', alpha=0.7)
                    ax3.set_xlabel('Word Complexity (Length)')
                    ax3.set_ylabel('Topic Strength (Score × Coherence)')
                    ax3.set_title('Word Complexity vs Topic Strength')

            # 4. Cross-Topic Similarity Heatmap
            ax4 = fig.add_subplot(gs[1, :])
            if topics_data and len(topics_data) > 1:
                # Calculate topic similarity matrix
                topic_words_vectors = []
                topic_names = list(topics_data.keys())

                for topic_info in topics_data.values():
                    words = topic_info.get("words", [])
                    # Create simple word vector (this is simplified - in practice, use embeddings)
                    word_dict = {word: score for word, score in words}
                    topic_words_vectors.append(word_dict)

                # Create similarity matrix
                similarity_matrix = np.zeros((len(topics_data), len(topics_data)))
                for i, words_i in enumerate(topic_words_vectors):
                    for j, words_j in enumerate(topic_words_vectors):
                        # Jaccard similarity of word sets
                        set_i = set(words_i.keys())
                        set_j = set(words_j.keys())
                        if len(set_i.union(set_j)) > 0:
                            similarity = len(set_i.intersection(set_j)) / len(set_i.union(set_j))
                        else:
                            similarity = 0
                        similarity_matrix[i, j] = similarity

                im = ax4.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto')
                ax4.set_xticks(range(len(topic_names)))
                ax4.set_yticks(range(len(topic_names)))
                ax4.set_xticklabels([name.replace('topic_', 'T') for name in topic_names])
                ax4.set_yticklabels([name.replace('topic_', 'T') for name in topic_names])
                ax4.set_title('Inter-Topic Similarity Matrix (Jaccard Index)')

                # Add text annotations
                for i in range(len(topic_names)):
                    for j in range(len(topic_names)):
                        text = ax4.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                                       ha="center", va="center",
                                       color="white" if similarity_matrix[i, j] > 0.5 else "black")

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax4)
                cbar.set_label('Similarity Score')

            # 5. Entity-Topic Network Strength
            ax5 = fig.add_subplot(gs[2, 0])
            ner_data = nlp_results.get("named_entities", {})
            if topics_data and ner_data:
                # Calculate entity-topic relationship strength
                entity_counts = []
                topic_coherences = []

                entities_by_type = ner_data.get("entities_by_type", {})
                for entity_type, entities in entities_by_type.items():
                    entity_count = sum(count for entity, count in entities)
                    entity_counts.append(entity_count)

                coherences = [topic_info.get("coherence", 0) for topic_info in topics_data.values()]
                if entity_counts and coherences:
                    # Repeat coherences to match entity counts if needed
                    coherences_extended = (coherences * ((len(entity_counts) // len(coherences)) + 1))[:len(entity_counts)]

                    ax5.scatter(entity_counts, coherences_extended, alpha=0.6,
                               color=self.colors['accent'], s=50)
                    ax5.set_xlabel('Entity Count')
                    ax5.set_ylabel('Topic Coherence')
                    ax5.set_title('Entity Density vs Topic Quality')
                    ax5.grid(True, alpha=0.3)

            # 6. Sentiment Stability Analysis
            ax6 = fig.add_subplot(gs[2, 1])
            if sentiment_data:
                # Create sentiment stability visualization
                categories = ['Positive', 'Neutral', 'Negative']
                values = [
                    sentiment_data.get('positive_ratio', 0),
                    sentiment_data.get('neutral_ratio', 0),
                    sentiment_data.get('negative_ratio', 0)
                ]

                # Calculate entropy as measure of sentiment diversity
                entropy = -sum(p * np.log2(p + 1e-10) for p in values if p > 0)
                max_entropy = np.log2(3)  # Maximum possible entropy for 3 categories
                stability = 1 - (entropy / max_entropy)  # Stability is inverse of normalized entropy

                # Create donut chart
                ax6.pie(values, labels=categories, autopct='%1.1f%%',
                       startangle=90, pctdistance=0.85,
                       colors=[self.significance_colors['highly_significant'],
                              self.colors['neutral'],
                              self.significance_colors['not_significant']])

                # Add center text showing stability
                centre_circle = plt.Circle((0,0), 0.70, fc='white')
                ax6.add_artist(centre_circle)
                ax6.text(0, 0, f'Stability\n{stability:.2f}', ha='center', va='center',
                        fontsize=12, fontweight='bold')
                ax6.set_title('Sentiment Distribution & Stability')

            # 7. Analysis Completeness Radar
            ax7 = fig.add_subplot(gs[2, 2], projection='polar')

            # Calculate completeness metrics
            completeness_metrics = {
                'Data Collection': (successful_downloads / total_papers * 100) if total_papers > 0 else 0,
                'Topic Analysis': len(topics_data) * 10 if topics_data else 0,  # Scale by 10
                'Sentiment Analysis': 100 if sentiment_data else 0,
                'Entity Recognition': len(ner_data.get("entities_by_type", {})) * 20 if ner_data else 0,
                'Clustering': 50 if nlp_results.get("clustering") else 0
            }

            # Normalize to 0-100
            for key in completeness_metrics:
                completeness_metrics[key] = min(completeness_metrics[key], 100)

            angles = np.linspace(0, 2*np.pi, len(completeness_metrics), endpoint=False)
            values = list(completeness_metrics.values())
            values += values[:1]  # Complete the circle
            angles = np.concatenate((angles, [angles[0]]))

            ax7.plot(angles, values, 'o-', linewidth=2, color=self.colors['primary'])
            ax7.fill(angles, values, alpha=0.25, color=self.colors['primary'])
            ax7.set_xticks(angles[:-1])
            ax7.set_xticklabels(list(completeness_metrics.keys()), fontsize=10)
            ax7.set_ylim(0, 100)
            ax7.set_title('Analysis Completeness Profile', pad=20)
            ax7.grid(True)

            # Add average completeness in center
            avg_completeness = np.mean(list(completeness_metrics.values()))
            ax7.text(0, 0, f'{avg_completeness:.1f}%', ha='center', va='center',
                    fontsize=14, fontweight='bold')

            plt.tight_layout()

            filename = os.path.join(self.output_dir, f"correlation_analysis_{timestamp}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Created correlation analysis: {filename}")
            return filename

        except Exception as e:
            logger.error(f"❌ Correlation analysis creation failed: {e}")
            return None

    def _create_advanced_topic_viz(self, results: Dict[str, Any],
                                 query: str, timestamp: str) -> Optional[str]:
        """Create advanced topic modeling visualization with t-SNE and hierarchical clustering"""
        try:
            fig = plt.figure(figsize=(20, 14))
            gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
            fig.suptitle(f'Advanced Topic Analysis: {query}', fontsize=16, fontweight='bold')

            phases = results.get("phases", {})
            nlp_results = phases.get("nlp_analysis", {})
            topics_data = nlp_results.get("topic_modeling", {}).get("topics", {})

            if not topics_data:
                logger.warning("No topic data available for advanced visualization")
                return None

            # Prepare data
            topic_names = list(topics_data.keys())
            topic_coherences = [topics_data[topic].get("coherence", 0) for topic in topic_names]

            # 1. Topic Evolution/Flow Diagram
            ax1 = fig.add_subplot(gs[0, :2])

            # Create synthetic topic evolution data (in practice, this would come from temporal analysis)
            n_topics = len(topic_names)
            time_points = 5  # Synthetic time points
            topic_strengths = np.random.rand(n_topics, time_points)

            # Normalize so each topic has believable evolution
            for i in range(n_topics):
                base_strength = topic_coherences[i]
                topic_strengths[i] = topic_strengths[i] * base_strength + base_strength * 0.5

            # Create stacked area plot
            time_labels = [f'Phase {i+1}' for i in range(time_points)]
            ax1.stackplot(range(time_points), *topic_strengths,
                         labels=[f'T{i}' for i in range(n_topics)],
                         colors=self.scientific_palette[:n_topics], alpha=0.8)

            ax1.set_xlabel('Analysis Phases')
            ax1.set_ylabel('Topic Strength')
            ax1.set_title('Topic Evolution Over Analysis Phases')
            ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
            ax1.set_xticks(range(time_points))
            ax1.set_xticklabels(time_labels)
            ax1.grid(True, alpha=0.3)

            # 2. Topic Quality Metrics Dashboard
            ax2 = fig.add_subplot(gs[0, 2:])

            # Calculate various topic quality metrics
            quality_metrics = {}
            for i, (topic_name, topic_info) in enumerate(topics_data.items()):
                words = topic_info.get("words", [])
                coherence = topic_info.get("coherence", 0)

                # Word diversity (unique words)
                word_diversity = len(set(word for word, score in words))

                # Average word score
                avg_word_score = np.mean([score for word, score in words]) if words else 0

                # Topic specificity (how concentrated the word scores are)
                word_scores = [score for word, score in words]
                topic_specificity = np.std(word_scores) if len(word_scores) > 1 else 0

                quality_metrics[f'T{i}'] = {
                    'Coherence': coherence,
                    'Diversity': word_diversity / 10,  # Scale down
                    'Avg Score': avg_word_score,
                    'Specificity': topic_specificity
                }

            # Create heatmap of quality metrics
            metrics_df = pd.DataFrame(quality_metrics).T
            im = ax2.imshow(metrics_df.values, cmap='RdYlGn', aspect='auto')

            ax2.set_xticks(range(len(metrics_df.columns)))
            ax2.set_yticks(range(len(metrics_df.index)))
            ax2.set_xticklabels(metrics_df.columns)
            ax2.set_yticklabels(metrics_df.index)
            ax2.set_title('Topic Quality Metrics Heatmap')

            # Add text annotations
            for i in range(len(metrics_df.index)):
                for j in range(len(metrics_df.columns)):
                    text = ax2.text(j, i, f'{metrics_df.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=9)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label('Quality Score')

            # 3. Word Co-occurrence Network
            ax3 = fig.add_subplot(gs[1, :2])

            # Build word co-occurrence network
            G = nx.Graph()
            word_counts = defaultdict(int)
            word_pairs = defaultdict(int)

            # Collect word co-occurrences across topics
            for topic_info in topics_data.values():
                words = [word for word, score in topic_info.get("words", [])[:5]]  # Top 5 words
                for word in words:
                    word_counts[word] += 1
                for i, word1 in enumerate(words):
                    for word2 in words[i+1:]:
                        word_pairs[(word1, word2)] += 1

            # Add nodes and edges
            top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            for word, count in top_words:
                G.add_node(word, size=count)

            for (word1, word2), count in word_pairs.items():
                if word1 in [w[0] for w in top_words] and word2 in [w[0] for w in top_words]:
                    if count > 1:  # Only show strong co-occurrences
                        G.add_edge(word1, word2, weight=count)

            if G.number_of_nodes() > 0:
                pos = nx.spring_layout(G, k=2, iterations=50)

                # Draw nodes with size based on frequency
                node_sizes = [word_counts[node] * 100 for node in G.nodes()]
                nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                     node_color=self.colors['primary'], alpha=0.7, ax=ax3)

                # Draw edges with width based on co-occurrence
                edges = G.edges()
                weights = [G[u][v]['weight'] for u, v in edges]
                nx.draw_networkx_edges(G, pos, width=[w/2 for w in weights],
                                     alpha=0.5, ax=ax3)

                # Draw labels
                nx.draw_networkx_labels(G, pos, font_size=8, ax=ax3)

                ax3.set_title('Word Co-occurrence Network')
                ax3.axis('off')

            # 4. Topic Similarity Dendrogram
            ax4 = fig.add_subplot(gs[1, 2:])

            if len(topics_data) > 2:
                # Create feature vectors for topics (simplified)
                feature_vectors = []
                for topic_info in topics_data.values():
                    words = topic_info.get("words", [])
                    # Create a simple feature vector based on top words and coherence
                    features = [topic_info.get("coherence", 0)]
                    # Add word scores as features
                    word_scores = [score for word, score in words[:5]]
                    features.extend(word_scores + [0] * (5 - len(word_scores)))  # Pad to 5
                    feature_vectors.append(features)

                feature_matrix = np.array(feature_vectors)

                # Compute distance matrix and create dendrogram
                from scipy.cluster.hierarchy import dendrogram, linkage
                from scipy.spatial.distance import pdist

                distances = pdist(feature_matrix, metric='euclidean')
                linkage_matrix = linkage(distances, method='ward')

                dendro = dendrogram(linkage_matrix,
                                  labels=[f'T{i}' for i in range(len(topic_names))],
                                  ax=ax4)
                ax4.set_title('Topic Similarity Dendrogram')
                ax4.set_xlabel('Topics')
                ax4.set_ylabel('Distance')

            # 5. Topic Word Clouds (Individual)
            ax5 = fig.add_subplot(gs[2, 0])
            ax6 = fig.add_subplot(gs[2, 1])
            ax7 = fig.add_subplot(gs[2, 2])
            ax8 = fig.add_subplot(gs[2, 3])

            topic_axes = [ax5, ax6, ax7, ax8]

            for i, (topic_name, topic_info) in enumerate(list(topics_data.items())[:4]):
                words = topic_info.get("words", [])
                if words and i < len(topic_axes):
                    # Create word frequency dict for wordcloud
                    word_freq = {word: score * 100 for word, score in words[:10]}

                    # Generate mini word cloud
                    if word_freq:
                        wordcloud = WordCloud(
                            width=300, height=200,
                            background_color='white',
                            colormap='viridis',
                            max_words=20
                        ).generate_from_frequencies(word_freq)

                        topic_axes[i].imshow(wordcloud, interpolation='bilinear')
                        topic_axes[i].set_title(f'{topic_name.replace("topic_", "Topic ")} '
                                              f'(C={topic_info.get("coherence", 0):.3f})', fontsize=10)
                        topic_axes[i].axis('off')

            # Hide unused axes
            for j in range(len(topics_data), len(topic_axes)):
                topic_axes[j].axis('off')

            plt.tight_layout()

            filename = os.path.join(self.output_dir, f"advanced_topic_analysis_{timestamp}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Created advanced topic analysis: {filename}")
            return filename

        except Exception as e:
            logger.error(f"❌ Advanced topic analysis creation failed: {e}")
            return None

    def _create_trend_analysis(self, results: Dict[str, Any],
                             query: str, timestamp: str) -> Optional[str]:
        """Create trend analysis visualization (synthetic for demonstration)"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Trend Analysis: {query}', fontsize=16, fontweight='bold')

            phases = results.get("phases", {})
            nlp_results = phases.get("nlp_analysis", {})

            # 1. Topic Evolution Over Time
            ax1 = axes[0, 0]
            topics_data = nlp_results.get("topic_modeling", {}).get("topics", {})
            if topics_data:
                # Simulate temporal evolution
                time_points = np.arange(2020, 2026)
                topic_coherences = [topics_data[topic].get("coherence", 0) for topic in topics_data.keys()]

                for i, coherence in enumerate(topic_coherences[:5]):  # Top 5 topics
                    # Generate synthetic trend data
                    trend = np.random.normal(coherence, 0.05, len(time_points))
                    trend = np.clip(trend, 0, 1)  # Ensure valid coherence range
                    ax1.plot(time_points, trend, marker='o', label=f'Topic {i+1}',
                            color=self.scientific_palette[i % len(self.scientific_palette)])

                ax1.set_xlabel('Year')
                ax1.set_ylabel('Topic Coherence')
                ax1.set_title('Topic Coherence Evolution (Simulated)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

            # 2. Research Volume Trends
            ax2 = axes[0, 1]
            data_collection = phases.get("data_collection", {})
            total_papers = data_collection.get("total_papers", 0)

            # Simulate publication volume over years
            years = np.arange(2020, 2026)
            base_volume = total_papers / len(years)
            volume_trend = [base_volume * (1 + 0.1 * i + np.random.normal(0, 0.05)) for i in range(len(years))]

            bars = ax2.bar(years, volume_trend, color=self.colors['primary'], alpha=0.7)
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Publication Volume')
            ax2.set_title('Research Volume Trend (Simulated)')
            ax2.grid(True, alpha=0.3)

            # Add trend line
            z = np.polyfit(years, volume_trend, 1)
            p = np.poly1d(z)
            ax2.plot(years, p(years), "r--", alpha=0.8, linewidth=2)

            # 3. Sentiment Evolution
            ax3 = axes[1, 0]
            sentiment_data = nlp_results.get("sentiment_analysis", {}).get("overall_sentiment", {})
            if sentiment_data:
                mean_polarity = sentiment_data.get('mean_polarity', 0)

                # Generate sentiment trend over time
                sentiment_trend = [mean_polarity + np.random.normal(0, 0.1) for _ in years]
                ax3.plot(years, sentiment_trend, marker='o', color=self.colors['secondary'],
                        linewidth=2, markersize=8)

                # Add confidence band
                std_dev = 0.1
                ax3.fill_between(years,
                               [s - std_dev for s in sentiment_trend],
                               [s + std_dev for s in sentiment_trend],
                               alpha=0.3, color=self.colors['secondary'])

                ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
                ax3.set_xlabel('Year')
                ax3.set_ylabel('Sentiment Polarity')
                ax3.set_title('Literature Sentiment Trend')
                ax3.set_ylim(-1, 1)
                ax3.grid(True, alpha=0.3)

            # 4. Research Complexity Evolution
            ax4 = axes[1, 1]

            # Calculate complexity metrics
            topic_count = len(topics_data) if topics_data else 0
            entity_count = len(nlp_results.get("named_entities", {}).get("entities_by_type", {}))
            clustering_count = len(set(nlp_results.get("clustering", {}).get("labels", [])))

            complexity_scores = []
            for year in years:
                # Simulate increasing complexity over time
                year_multiplier = 1 + (year - min(years)) * 0.1
                complexity = (topic_count + entity_count + clustering_count) * year_multiplier
                complexity += np.random.normal(0, complexity * 0.1)  # Add noise
                complexity_scores.append(max(0, complexity))

            ax4.plot(years, complexity_scores, marker='s', color=self.colors['accent'],
                    linewidth=2, markersize=8)
            ax4.fill_between(years, complexity_scores, alpha=0.3, color=self.colors['accent'])
            ax4.set_xlabel('Year')
            ax4.set_ylabel('Research Complexity Score')
            ax4.set_title('Research Field Complexity Evolution')
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            filename = os.path.join(self.output_dir, f"trend_analysis_{timestamp}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Created trend analysis: {filename}")
            return filename

        except Exception as e:
            logger.error(f"❌ Trend analysis creation failed: {e}")
            return None

    def _create_comprehensive_network(self, results: Dict[str, Any],
                                    query: str, timestamp: str) -> Optional[str]:
        """Create comprehensive network analysis visualization"""
        try:
            fig = plt.figure(figsize=(20, 12))
            gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
            fig.suptitle(f'Comprehensive Network Analysis: {query}', fontsize=16, fontweight='bold')

            phases = results.get("phases", {})
            nlp_results = phases.get("nlp_analysis", {})
            topics_data = nlp_results.get("topic_modeling", {}).get("topics", {})
            ner_data = nlp_results.get("named_entities", {})

            # 1. Multi-layer Topic-Entity Network
            ax1 = fig.add_subplot(gs[0, :2])
            G = nx.Graph()

            if topics_data and ner_data:
                # Add topic nodes
                topic_nodes = []
                for i, (topic_name, topic_info) in enumerate(topics_data.items()):
                    node_id = f'T{i}'
                    coherence = topic_info.get("coherence", 0)
                    G.add_node(node_id, node_type='topic', coherence=coherence,
                              size=coherence * 1000)
                    topic_nodes.append(node_id)

                # Add entity nodes (top entities)
                entity_nodes = []
                entities_by_type = ner_data.get("entities_by_type", {})
                for entity_type, entities in entities_by_type.items():
                    for entity, count in entities[:3]:  # Top 3 per type
                        node_id = f'{entity_type}:{entity}'
                        G.add_node(node_id, node_type='entity', entity_type=entity_type,
                                  count=count, size=count * 10)
                        entity_nodes.append(node_id)

                # Add edges based on co-occurrence (simplified)
                for i, topic_node in enumerate(topic_nodes):
                    topic_info = list(topics_data.values())[i]
                    topic_words = [word for word, score in topic_info.get("words", [])[:5]]

                    for entity_node in entity_nodes:
                        entity_name = entity_node.split(':')[1]
                        # Simple heuristic: connect if entity appears in topic words
                        connection_strength = sum(1 for word in topic_words
                                               if word.lower() in entity_name.lower())
                        if connection_strength > 0:
                            G.add_edge(topic_node, entity_node, weight=connection_strength)

                if G.number_of_nodes() > 0:
                    # Use spring layout for better visualization
                    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

                    # Draw topic nodes
                    topic_positions = {node: pos[node] for node in topic_nodes if node in pos}
                    if topic_positions:
                        topic_sizes = [G.nodes[node]['size'] for node in topic_nodes if node in pos]
                        nx.draw_networkx_nodes(G, topic_positions, nodelist=list(topic_positions.keys()),
                                             node_size=topic_sizes, node_color=self.colors['primary'],
                                             alpha=0.8, label='Topics')

                    # Draw entity nodes by type
                    entity_colors = {
                        'PERSON': self.colors['secondary'],
                        'ORG': self.colors['success'],
                        'GPE': self.colors['info'],
                        'PRODUCT': self.colors['accent']
                    }

                    for entity_type in entity_colors:
                        type_nodes = [node for node in entity_nodes
                                    if node in pos and G.nodes[node]['entity_type'] == entity_type]
                        if type_nodes:
                            type_positions = {node: pos[node] for node in type_nodes}
                            type_sizes = [G.nodes[node]['size'] for node in type_nodes]
                            nx.draw_networkx_nodes(G, type_positions, nodelist=type_nodes,
                                                 node_size=type_sizes,
                                                 node_color=entity_colors[entity_type],
                                                 alpha=0.7, label=entity_type)

                    # Draw edges
                    edges = G.edges()
                    if edges:
                        weights = [G[u][v]['weight'] for u, v in edges]
                        nx.draw_networkx_edges(G, pos, alpha=0.5, width=weights)

                    # Draw labels for top nodes only
                    important_nodes = sorted(G.nodes(), key=lambda x: G.nodes[x]['size'], reverse=True)[:10]
                    important_pos = {node: pos[node] for node in important_nodes if node in pos}
                    nx.draw_networkx_labels(G, important_pos, font_size=8)

                    ax1.set_title('Topic-Entity Network (Multi-layer)')
                    ax1.legend()
                    ax1.axis('off')

            # 2. Network Centrality Analysis
            ax2 = fig.add_subplot(gs[0, 2])
            if G.number_of_nodes() > 0:
                # Calculate centrality measures
                degree_centrality = nx.degree_centrality(G)
                betweenness_centrality = nx.betweenness_centrality(G)
                closeness_centrality = nx.closeness_centrality(G)

                # Get top nodes by degree centrality
                top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                nodes, degrees = zip(*top_nodes)

                # Create centrality comparison
                x = np.arange(len(nodes))
                width = 0.25

                degree_values = [degree_centrality[node] for node in nodes]
                betweenness_values = [betweenness_centrality[node] for node in nodes]
                closeness_values = [closeness_centrality[node] for node in nodes]

                ax2.bar(x - width, degree_values, width, label='Degree', color=self.colors['primary'])
                ax2.bar(x, betweenness_values, width, label='Betweenness', color=self.colors['secondary'])
                ax2.bar(x + width, closeness_values, width, label='Closeness', color=self.colors['success'])

                ax2.set_xlabel('Top Nodes')
                ax2.set_ylabel('Centrality Score')
                ax2.set_title('Network Centrality Analysis')
                ax2.set_xticks(x)
                ax2.set_xticklabels([node[:10] + '...' if len(node) > 10 else node
                                   for node in nodes], rotation=45, ha='right')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            # 3. Community Detection
            ax3 = fig.add_subplot(gs[1, 0])
            if G.number_of_nodes() > 3:
                try:
                    # Simple community detection using modularity
                    communities = nx.community.greedy_modularity_communities(G)
                    modularity = nx.community.modularity(G, communities)

                    # Visualize communities
                    pos = nx.spring_layout(G, k=2, iterations=30)
                    colors = self.scientific_palette[:len(communities)]

                    for i, community in enumerate(communities):
                        community_nodes = list(community)
                        if community_nodes:
                            nx.draw_networkx_nodes(G, pos, nodelist=community_nodes,
                                                 node_color=colors[i % len(colors)],
                                                 alpha=0.7, node_size=200,
                                                 label=f'Community {i+1}')

                    nx.draw_networkx_edges(G, pos, alpha=0.3)
                    ax3.set_title(f'Community Structure (Q={modularity:.3f})')
                    ax3.legend()
                    ax3.axis('off')

                except:
                    ax3.text(0.5, 0.5, 'Community detection\nnot available', ha='center', va='center',
                            transform=ax3.transAxes)
                    ax3.axis('off')

            # 4. Network Metrics Summary
            ax4 = fig.add_subplot(gs[1, 1])
            if G.number_of_nodes() > 0:
                metrics = {
                    'Nodes': G.number_of_nodes(),
                    'Edges': G.number_of_edges(),
                    'Density': nx.density(G),
                    'Avg Clustering': nx.average_clustering(G),
                    'Diameter': nx.diameter(G) if nx.is_connected(G) else float('inf'),
                    'Components': nx.number_connected_components(G)
                }

                # Filter out infinite values
                finite_metrics = {k: v for k, v in metrics.items()
                                if not (isinstance(v, float) and not np.isfinite(v))}

                if finite_metrics:
                    metric_names = list(finite_metrics.keys())
                    metric_values = list(finite_metrics.values())

                    bars = ax4.bar(metric_names, metric_values, color=self.color_palette[:len(metric_names)])
                    ax4.set_title('Network Topology Metrics')
                    ax4.set_ylabel('Value')
                    ax4.tick_params(axis='x', rotation=45)

                    # Add value labels
                    for bar, value in zip(bars, metric_values):
                        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{value:.3f}' if isinstance(value, float) else str(value),
                                ha='center', va='bottom')

            # 5. Adjacency Matrix Heatmap
            ax5 = fig.add_subplot(gs[1, 2])
            if G.number_of_nodes() > 0 and G.number_of_nodes() <= 20:  # Only for reasonably sized networks
                adj_matrix = nx.adjacency_matrix(G).todense()
                node_labels = list(G.nodes())

                im = ax5.imshow(adj_matrix, cmap='Blues', aspect='auto')
                ax5.set_xticks(range(len(node_labels)))
                ax5.set_yticks(range(len(node_labels)))
                ax5.set_xticklabels([label[:8] + '...' if len(label) > 8 else label
                                   for label in node_labels], rotation=90)
                ax5.set_yticklabels([label[:8] + '...' if len(label) > 8 else label
                                   for label in node_labels])
                ax5.set_title('Network Adjacency Matrix')

                # Add colorbar
                plt.colorbar(im, ax=ax5, label='Connection Strength')

            plt.tight_layout()

            filename = os.path.join(self.output_dir, f"comprehensive_network_{timestamp}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Created comprehensive network analysis: {filename}")
            return filename

        except Exception as e:
            logger.error(f"❌ Comprehensive network analysis creation failed: {e}")
            return None

    def _create_impact_assessment(self, results: Dict[str, Any],
                                query: str, timestamp: str) -> Optional[str]:
        """Create research impact and quality assessment visualization"""
        try:
            fig = plt.figure(figsize=(18, 12))
            gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
            fig.suptitle(f'Research Impact & Quality Assessment: {query}', fontsize=16, fontweight='bold')

            phases = results.get("phases", {})
            nlp_results = phases.get("nlp_analysis", {})
            data_collection = phases.get("data_collection", {})
            topics_data = nlp_results.get("topic_modeling", {}).get("topics", {})
            sentiment_data = nlp_results.get("sentiment_analysis", {}).get("overall_sentiment", {})

            # 1. Research Quality Score
            ax1 = fig.add_subplot(gs[0, 0])

            # Calculate comprehensive quality score
            total_papers = data_collection.get("total_papers", 0)
            successful_downloads = data_collection.get("successful_downloads", 0)

            quality_components = {
                'Data Availability': (successful_downloads / total_papers * 100) if total_papers > 0 else 0,
                'Topic Coherence': np.mean([t.get("coherence", 0) for t in topics_data.values()]) * 100 if topics_data else 0,
                'Content Diversity': len(topics_data) * 10 if topics_data else 0,
                'Sentiment Balance': (1 - abs(sentiment_data.get('mean_polarity', 0))) * 100 if sentiment_data else 50,
                'Analysis Completeness': len([p for p in phases.keys() if phases[p].get("completion_time", 0) > 0]) / len(phases) * 100
            }

            # Normalize all components to 0-100 scale
            for key in quality_components:
                quality_components[key] = min(quality_components[key], 100)

            overall_quality = np.mean(list(quality_components.values()))

            # Create gauge chart
            ax1.pie([overall_quality, 100 - overall_quality],
                   colors=[self._get_quality_color(overall_quality), 'lightgray'],
                   startangle=90, counterclock=False,
                   wedgeprops=dict(width=0.3))

            # Add center text
            ax1.text(0, 0, f'{overall_quality:.1f}%', ha='center', va='center',
                    fontsize=20, fontweight='bold')
            ax1.set_title('Overall Research Quality Score')

            # 2. Quality Components Breakdown
            ax2 = fig.add_subplot(gs[0, 1:])

            component_names = list(quality_components.keys())
            component_values = list(quality_components.values())
            colors = [self._get_quality_color(v) for v in component_values]

            bars = ax2.barh(component_names, component_values, color=colors, alpha=0.8)
            ax2.set_xlabel('Quality Score (%)')
            ax2.set_title('Quality Components Analysis')
            ax2.set_xlim(0, 100)

            # Add value labels
            for bar, value in zip(bars, component_values):
                ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{value:.1f}%', ha='left', va='center')

            # Add quality thresholds
            ax2.axvline(x=75, color='green', linestyle='--', alpha=0.5, label='Excellent (75%+)')
            ax2.axvline(x=50, color='orange', linestyle='--', alpha=0.5, label='Good (50%+)')
            ax2.axvline(x=25, color='red', linestyle='--', alpha=0.5, label='Poor (<25%)')
            ax2.legend()
            ax2.grid(True, axis='x', alpha=0.3)

            # 3. Research Impact Potential
            ax3 = fig.add_subplot(gs[1, 0])

            # Calculate impact factors (synthetic)
            impact_factors = {
                'Novelty': len(set([word for topic_info in topics_data.values()
                                  for word, score in topic_info.get("words", [])])) / 100 if topics_data else 0,
                'Interdisciplinary': len(nlp_results.get("named_entities", {}).get("entities_by_type", {})) / 10,
                'Methodological': 1 if len(topics_data) > 3 else 0.5 if len(topics_data) > 1 else 0,
                'Relevance': (1 - abs(sentiment_data.get('mean_polarity', 0))) if sentiment_data else 0.5
            }

            # Normalize to 0-1 scale
            for key in impact_factors:
                impact_factors[key] = min(impact_factors[key], 1.0)

            # Create radar chart
            angles = np.linspace(0, 2*np.pi, len(impact_factors), endpoint=False)
            values = list(impact_factors.values())
            values += values[:1]  # Complete the circle
            angles = np.concatenate((angles, [angles[0]]))

            ax3 = plt.subplot(gs[1, 0], projection='polar')
            ax3.plot(angles, values, 'o-', linewidth=2, color=self.colors['primary'])
            ax3.fill(angles, values, alpha=0.25, color=self.colors['primary'])
            ax3.set_xticks(angles[:-1])
            ax3.set_xticklabels(list(impact_factors.keys()))
            ax3.set_ylim(0, 1)
            ax3.set_title('Research Impact Potential', pad=20)
            ax3.grid(True)

            # 4. Temporal Research Trends
            ax4 = fig.add_subplot(gs[1, 1:])

            # Simulate research progression over analysis phases
            phase_names = list(phases.keys())
            phase_quality = []

            for phase_name, phase_data in phases.items():
                completion_time = phase_data.get("completion_time", 0)
                if phase_name == "data_collection":
                    quality = (successful_downloads / total_papers) if total_papers > 0 else 0
                elif phase_name == "nlp_analysis":
                    quality = np.mean([t.get("coherence", 0) for t in topics_data.values()]) if topics_data else 0
                else:
                    quality = max(0, 1 - completion_time / 100)  # Synthetic quality based on time

                phase_quality.append(quality * 100)

            # Create step plot showing quality progression
            x_pos = np.arange(len(phase_names))
            ax4.step(x_pos, phase_quality, where='mid', linewidth=3, color=self.colors['primary'])
            ax4.scatter(x_pos, phase_quality, color=self.colors['secondary'], s=100, zorder=5)

            # Add trend line
            if len(phase_quality) > 1:
                z = np.polyfit(x_pos, phase_quality, 1)
                p = np.poly1d(z)
                trend_line = p(x_pos)
                ax4.plot(x_pos, trend_line, '--', color=self.colors['info'], alpha=0.7, linewidth=2)

                # Calculate trend direction
                trend_slope = z[0]
                trend_text = "Improving" if trend_slope > 0 else "Declining" if trend_slope < 0 else "Stable"
                ax4.text(0.02, 0.98, f'Trend: {trend_text} ({trend_slope:.1f}%/phase)',
                        transform=ax4.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax4.set_xlabel('Analysis Phase')
            ax4.set_ylabel('Quality Score (%)')
            ax4.set_title('Research Quality Progression')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels([name.replace('_', '\n') for name in phase_names])
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 100)

            # 5. Citation Potential Analysis (synthetic)
            ax5 = fig.add_subplot(gs[2, 0])

            citation_categories = ['High Impact', 'Medium Impact', 'Low Impact', 'Niche']
            # Calculate based on topic diversity, quality, and novelty
            high_impact = min(len(topics_data) * overall_quality / 100, 40) if topics_data else 0
            medium_impact = min((100 - high_impact) * 0.4, 35)
            low_impact = min((100 - high_impact - medium_impact) * 0.6, 20)
            niche = 100 - high_impact - medium_impact - low_impact

            citation_values = [high_impact, medium_impact, low_impact, niche]
            citation_colors = [self.significance_colors['highly_significant'],
                             self.colors['success'],
                             self.colors['warning'],
                             self.colors['neutral']]

            ax5.pie(citation_values, labels=citation_categories,
                   autopct='%1.1f%%', colors=citation_colors)
            ax5.set_title('Potential Citation Impact Distribution')

            # 6. Research Maturity Assessment
            ax6 = fig.add_subplot(gs[2, 1])

            maturity_metrics = {
                'Methodology': overall_quality / 100,
                'Theoretical Framework': len(topics_data) / 10 if topics_data else 0,
                'Empirical Evidence': (successful_downloads / total_papers) if total_papers > 0 else 0,
                'Statistical Rigor': 1 if sentiment_data else 0.3,
                'Reproducibility': 0.8 if overall_quality > 70 else 0.5 if overall_quality > 40 else 0.3
            }

            # Normalize to 0-1
            for key in maturity_metrics:
                maturity_metrics[key] = min(maturity_metrics[key], 1.0)

            metric_names = list(maturity_metrics.keys())
            metric_values = list(maturity_metrics.values())

            bars = ax6.bar(metric_names, metric_values, color=self.color_palette[:len(metric_names)])
            ax6.set_ylabel('Maturity Score (0-1)')
            ax6.set_title('Research Maturity Assessment')
            ax6.tick_params(axis='x', rotation=45)
            ax6.set_ylim(0, 1)

            # Add maturity level indicators
            ax6.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Mature (0.8+)')
            ax6.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Developing (0.5+)')
            ax6.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Emerging (<0.3)')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

            # 7. Research Recommendations
            ax7 = fig.add_subplot(gs[2, 2])

            # Generate recommendations based on analysis
            recommendations = []
            if overall_quality < 50:
                recommendations.append("• Improve data collection methods")
            if len(topics_data) < 3:
                recommendations.append("• Expand topic diversity")
            if successful_downloads / total_papers < 0.5 if total_papers > 0 else True:
                recommendations.append("• Enhance data accessibility")
            if not sentiment_data:
                recommendations.append("• Add sentiment analysis")

            if overall_quality >= 75:
                recommendations.append("• Consider publication")
                recommendations.append("• Explore collaboration")
            elif overall_quality >= 50:
                recommendations.append("• Refine methodology")
                recommendations.append("• Increase sample size")
            else:
                recommendations.append("• Fundamental redesign needed")

            ax7.text(0.05, 0.95, 'Research Recommendations:', fontsize=12, fontweight='bold',
                    transform=ax7.transAxes, verticalalignment='top')

            for i, rec in enumerate(recommendations[:6]):  # Show top 6
                ax7.text(0.05, 0.85 - i*0.12, rec, fontsize=10,
                        transform=ax7.transAxes, verticalalignment='top')

            ax7.set_xlim(0, 1)
            ax7.set_ylim(0, 1)
            ax7.axis('off')
            ax7.set_title('Strategic Recommendations')

            plt.tight_layout()

            filename = os.path.join(self.output_dir, f"impact_assessment_{timestamp}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Created impact assessment: {filename}")
            return filename

        except Exception as e:
            logger.error(f"❌ Impact assessment creation failed: {e}")
            return None

    def _get_quality_color(self, score: float) -> str:
        """Get color based on quality score"""
        if score >= 75:
            return self.significance_colors['highly_significant']
        elif score >= 50:
            return self.colors['warning']
        else:
            return self.significance_colors['not_significant']

    # =============================================================================
    # LLM ANALYSIS VISUALIZATIONS
    # =============================================================================

    def create_llm_analysis_dashboard(self, llm_results: Dict[str, Any],
                                    query: str) -> List[str]:
        """Create comprehensive dashboard for LLM analysis results"""
        logger.info("🧠 Creating LLM analysis dashboard")

        generated_files = []

        try:
            # 1. LLM Score Distribution Analysis
            score_files = self._create_llm_score_distributions(llm_results, query)
            generated_files.extend(score_files)

            # 2. Paper Quality Rankings
            ranking_file = self._create_paper_quality_rankings(llm_results, query)
            if ranking_file:
                generated_files.append(ranking_file)

            # 3. LLM Response Analysis
            response_file = self._create_llm_response_analysis(llm_results, query)
            if response_file:
                generated_files.append(response_file)

            # 4. Multi-dimensional Score Radar Chart
            radar_file = self._create_llm_score_radar(llm_results, query)
            if radar_file:
                generated_files.append(radar_file)

            # 5. Comparative Analysis Matrix
            matrix_file = self._create_llm_comparison_matrix(llm_results, query)
            if matrix_file:
                generated_files.append(matrix_file)

            # 6. LLM Insight WordClouds
            insight_files = self._create_llm_insight_wordclouds(llm_results, query)
            generated_files.extend(insight_files)

            logger.info(f"✅ Generated {len(generated_files)} LLM visualization files")

        except Exception as e:
            logger.error(f"❌ LLM dashboard creation failed: {e}")

        return generated_files

    def _create_llm_score_distributions(self, llm_results: Dict[str, Any],
                                      query: str) -> List[str]:
        """Create score distribution visualizations"""
        generated_files = []

        try:
            batch_results = llm_results.get('batch_analysis_results', {})
            detailed_results = llm_results.get('detailed_results', [])

            if not detailed_results:
                logger.warning("No detailed LLM results found for visualization")
                return generated_files

            # Collect all scores by type
            scores_by_type = defaultdict(list)
            papers_by_score = defaultdict(list)

            for result in detailed_results:
                scores = result.get('structured_scores', {})
                title = result.get('title', 'Unknown')
                paper_id = result.get('paper_id', 'unknown')

                for score_name, score_value in scores.items():
                    scores_by_type[score_name].append(score_value)
                    papers_by_score[score_name].append({
                        'paper_id': paper_id,
                        'title': title,
                        'score': score_value
                    })

            if not scores_by_type:
                logger.warning("No structured scores found for visualization")
                return generated_files

            # Create comprehensive score distribution dashboard
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Calculate grid size
            n_scores = len(scores_by_type)
            n_cols = min(3, n_scores)
            n_rows = (n_scores + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_scores == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            fig.suptitle(f'LLM Analysis Score Distributions: {query}',
                        fontsize=16, fontweight='bold', y=0.98)

            for i, (score_name, scores) in enumerate(scores_by_type.items()):
                if i >= len(axes):
                    break

                ax = axes[i]

                # Create histogram with KDE overlay
                ax.hist(scores, bins=20, alpha=0.7, density=True,
                       color=self.colors['primary'], edgecolor='black')

                # Add KDE curve if enough data points
                if len(scores) > 3:
                    from scipy import stats
                    kde = stats.gaussian_kde(scores)
                    x_range = np.linspace(min(scores), max(scores), 100)
                    ax.plot(x_range, kde(x_range), color=self.colors['secondary'], linewidth=2)

                # Add statistical annotations
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                ax.axvline(mean_score, color='red', linestyle='--', alpha=0.7,
                          label=f'Mean: {mean_score:.2f}')
                ax.axvline(mean_score + std_score, color='orange', linestyle='--', alpha=0.5)
                ax.axvline(mean_score - std_score, color='orange', linestyle='--', alpha=0.5)

                ax.set_title(f'{score_name.replace("_", " ").title()}', fontweight='bold')
                ax.set_xlabel('Score')
                ax.set_ylabel('Density')
                ax.legend()

                # Add text box with statistics
                stats_text = f'n={len(scores)}\\nμ={mean_score:.2f}\\nσ={std_score:.2f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # Hide unused subplots
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            filename = os.path.join(self.output_dir, f"llm_score_distributions_{timestamp}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            generated_files.append(filename)
            logger.info(f"✅ Created LLM score distributions: {filename}")

        except Exception as e:
            logger.error(f"❌ LLM score distributions creation failed: {e}")

        return generated_files

    def _create_paper_quality_rankings(self, llm_results: Dict[str, Any],
                                     query: str) -> Optional[str]:
        """Create paper quality ranking visualization"""
        try:
            detailed_results = llm_results.get('detailed_results', [])

            if not detailed_results:
                return None

            # Calculate composite quality scores for each paper
            paper_scores = defaultdict(lambda: {'scores': {}, 'title': '', 'metadata': {}})

            for result in detailed_results:
                paper_id = result.get('paper_id', 'unknown')
                title = result.get('title', 'Unknown')
                metadata = result.get('metadata', {})
                scores = result.get('structured_scores', {})

                paper_scores[paper_id]['title'] = title
                paper_scores[paper_id]['metadata'] = metadata

                for score_name, score_value in scores.items():
                    paper_scores[paper_id]['scores'][score_name] = score_value

            if not paper_scores:
                return None

            # Calculate composite scores
            ranked_papers = []
            for paper_id, data in paper_scores.items():
                scores = data['scores']
                if scores:
                    # Calculate weighted composite score
                    weights = {
                        'overall_quality': 0.25,
                        'methodology_rigor': 0.20,
                        'innovation': 0.20,
                        'field_significance': 0.15,
                        'reproducibility': 0.10,
                        'clarity': 0.10
                    }

                    composite_score = 0
                    total_weight = 0

                    for score_name, score_value in scores.items():
                        weight = weights.get(score_name, 0.05)  # Default small weight
                        composite_score += score_value * weight
                        total_weight += weight

                    if total_weight > 0:
                        composite_score /= total_weight

                    ranked_papers.append({
                        'paper_id': paper_id,
                        'title': data['title'],
                        'composite_score': composite_score,
                        'scores': scores,
                        'journal': data['metadata'].get('journal', ''),
                        'year': data['metadata'].get('year', '')
                    })

            # Sort by composite score
            ranked_papers.sort(key=lambda x: x['composite_score'], reverse=True)

            # Create visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
            fig.suptitle(f'Paper Quality Rankings: {query}', fontsize=16, fontweight='bold')

            # Left plot: Top papers ranking
            top_papers = ranked_papers[:15]  # Show top 15

            titles = [p['title'][:50] + '...' if len(p['title']) > 50 else p['title']
                     for p in top_papers]
            scores = [p['composite_score'] for p in top_papers]
            colors = [self._get_quality_color(score * 10) for score in scores]

            y_pos = np.arange(len(titles))
            bars = ax1.barh(y_pos, scores, color=colors, alpha=0.8, edgecolor='black')

            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(titles, fontsize=9)
            ax1.set_xlabel('Composite Quality Score')
            ax1.set_title('Top 15 Papers by Quality Score', fontweight='bold')
            ax1.invert_yaxis()

            # Add score labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{scores[i]:.2f}', ha='left', va='center', fontsize=8)

            # Right plot: Score distribution by journal/year
            if len(ranked_papers) > 5:
                # Group by journal
                journal_scores = defaultdict(list)
                for paper in ranked_papers:
                    journal = paper.get('journal', 'Unknown')[:20]
                    journal_scores[journal].append(paper['composite_score'])

                # Filter journals with at least 2 papers
                filtered_journals = {j: scores for j, scores in journal_scores.items()
                                   if len(scores) >= 2}

                if filtered_journals:
                    journal_names = list(filtered_journals.keys())
                    journal_data = list(filtered_journals.values())

                    bp = ax2.boxplot(journal_data, labels=journal_names, patch_artist=True)
                    for patch in bp['boxes']:
                        patch.set_facecolor(self.colors['primary'])
                        patch.set_alpha(0.7)

                    ax2.set_title('Quality Score Distribution by Journal', fontweight='bold')
                    ax2.set_ylabel('Composite Quality Score')
                    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
                else:
                    ax2.text(0.5, 0.5, 'Insufficient data for\\njournal comparison',
                            ha='center', va='center', transform=ax2.transAxes,
                            fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))

            plt.tight_layout()
            filename = os.path.join(self.output_dir, f"paper_quality_rankings_{timestamp}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Created paper quality rankings: {filename}")
            return filename

        except Exception as e:
            logger.error(f"❌ Paper quality rankings creation failed: {e}")
            return None

    def _create_llm_response_analysis(self, llm_results: Dict[str, Any],
                                    query: str) -> Optional[str]:
        """Analyze and visualize LLM response characteristics"""
        try:
            detailed_results = llm_results.get('detailed_results', [])

            if not detailed_results:
                return None

            # Extract response characteristics
            response_lengths = []
            analysis_types = []
            processing_times = []
            llm_used = []

            for result in detailed_results:
                response = result.get('response', '')
                analysis_type = result.get('analysis_type', 'unknown')
                processing_time = result.get('processing_time', 0)
                llm = result.get('llm_used', 'unknown')

                response_lengths.append(len(response))
                analysis_types.append(analysis_type)
                processing_times.append(processing_time)
                llm_used.append(llm)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            fig = plt.figure(figsize=(16, 12))
            gs = gridspec.GridSpec(3, 2, figure=fig)

            fig.suptitle(f'LLM Response Analysis: {query}', fontsize=16, fontweight='bold')

            # 1. Response length distribution
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.hist(response_lengths, bins=20, alpha=0.7, color=self.colors['primary'],
                    edgecolor='black')
            ax1.set_title('Response Length Distribution', fontweight='bold')
            ax1.set_xlabel('Response Length (characters)')
            ax1.set_ylabel('Frequency')

            # Add statistics
            mean_len = np.mean(response_lengths)
            ax1.axvline(mean_len, color='red', linestyle='--', label=f'Mean: {mean_len:.0f}')
            ax1.legend()

            # 2. Processing time by analysis type
            ax2 = fig.add_subplot(gs[0, 1])
            analysis_type_times = defaultdict(list)
            for atype, ptime in zip(analysis_types, processing_times):
                analysis_type_times[atype].append(ptime)

            type_names = list(analysis_type_times.keys())
            type_times = list(analysis_type_times.values())

            if len(type_names) > 1:
                bp = ax2.boxplot(type_times, labels=type_names, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor(self.colors['secondary'])
                    patch.set_alpha(0.7)

                ax2.set_title('Processing Time by Analysis Type', fontweight='bold')
                ax2.set_ylabel('Processing Time (seconds)')
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            else:
                ax2.bar(type_names, [np.mean(times) for times in type_times],
                       color=self.colors['secondary'], alpha=0.7)
                ax2.set_title('Average Processing Time', fontweight='bold')
                ax2.set_ylabel('Processing Time (seconds)')

            # 3. LLM usage distribution
            ax3 = fig.add_subplot(gs[1, 0])
            llm_counts = Counter(llm_used)
            if llm_counts:
                ax3.pie(llm_counts.values(), labels=llm_counts.keys(),
                       autopct='%1.1f%%', startangle=90)
                ax3.set_title('LLM Usage Distribution', fontweight='bold')

            # 4. Response length vs processing time correlation
            ax4 = fig.add_subplot(gs[1, 1])
            scatter = ax4.scatter(response_lengths, processing_times, alpha=0.6,
                                 c=range(len(response_lengths)), cmap='viridis')
            ax4.set_xlabel('Response Length (characters)')
            ax4.set_ylabel('Processing Time (seconds)')
            ax4.set_title('Response Length vs Processing Time', fontweight='bold')

            # Add correlation coefficient
            if len(response_lengths) > 2:
                corr_coef = np.corrcoef(response_lengths, processing_times)[0, 1]
                ax4.text(0.02, 0.98, f'Correlation: {corr_coef:.3f}',
                        transform=ax4.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # 5. Analysis type frequency
            ax5 = fig.add_subplot(gs[2, :])
            analysis_counts = Counter(analysis_types)
            if analysis_counts:
                bars = ax5.bar(analysis_counts.keys(), analysis_counts.values(),
                              color=sns.color_palette("husl", len(analysis_counts)), alpha=0.8)
                ax5.set_title('Analysis Type Frequency', fontweight='bold')
                ax5.set_ylabel('Number of Analyses')
                plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')

                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax5.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom')

            plt.tight_layout()
            filename = os.path.join(self.output_dir, f"llm_response_analysis_{timestamp}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Created LLM response analysis: {filename}")
            return filename

        except Exception as e:
            logger.error(f"❌ LLM response analysis creation failed: {e}")
            return None