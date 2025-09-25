#!/usr/bin/env python3
"""
Abstract-Optimized Visualization Module
Visualizations that work well with abstract-only data
"""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
import networkx as nx
from wordcloud import WordCloud
from datetime import datetime
import os
import re
try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

    def flesch_reading_ease(text):
        """Fallback implementation"""
        words = len(text.split())
        sentences = len([s for s in text.split('.') if s.strip()])
        if sentences == 0 or words == 0:
            return 50  # Default score
        avg_sentence_length = words / sentences
        return max(0, min(100, 206.835 - (1.015 * avg_sentence_length)))
import matplotlib.patches as patches

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


class AbstractOptimizedVisualizer:
    """Visualizer optimized for abstract-only data"""

    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Enhanced color palette
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'warning': '#F4A261',
            'info': '#264653'
        }

        logger.info(f"🎨 Abstract-Optimized Visualizer initialized - output: {output_dir}")

    def create_abstract_dashboard(self, results: Dict[str, Any], query: str) -> List[str]:
        """Create comprehensive dashboard optimized for abstract data"""
        logger.info("📊 Creating abstract-optimized dashboard")

        generated_files = []

        try:
            papers = results.get("papers", [])
            if not papers:
                logger.warning("No papers found for visualization")
                return generated_files

            # 1. Abstract Analysis Dashboard
            abstract_file = self._create_abstract_analysis_dashboard(papers, query)
            if abstract_file:
                generated_files.append(abstract_file)

            # 2. Journal and Year Analysis
            journal_file = self._create_journal_year_analysis(papers, query)
            if journal_file:
                generated_files.append(journal_file)

            # 3. Abstract Text Analytics
            text_file = self._create_text_analytics_dashboard(papers, query)
            if text_file:
                generated_files.append(text_file)

            # 4. Keyword and Term Analysis
            keyword_file = self._create_keyword_analysis(papers, query)
            if keyword_file:
                generated_files.append(keyword_file)

            # 5. Abstract Similarity Network
            network_file = self._create_abstract_similarity_network(papers, query)
            if network_file:
                generated_files.append(network_file)

            # 6. Comprehensive Summary
            summary_file = self._create_comprehensive_summary(papers, query)
            if summary_file:
                generated_files.append(summary_file)

            logger.info(f"✅ Generated {len(generated_files)} abstract-optimized visualizations")

        except Exception as e:
            logger.error(f"❌ Abstract dashboard creation failed: {e}")

        return generated_files

    def _create_abstract_analysis_dashboard(self, papers: List, query: str) -> Optional[str]:
        """Create abstract-specific analysis dashboard"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Abstract Analysis Dashboard: {query}', fontsize=16, fontweight='bold')

            # 1. Abstract lengths distribution
            abstract_lengths = []
            for paper in papers:
                abstract = getattr(paper, 'abstract', '') or ''
                abstract_lengths.append(len(abstract.split()))

            if abstract_lengths:
                ax1.hist(abstract_lengths, bins=min(15, len(set(abstract_lengths))),
                        alpha=0.7, color=self.colors['primary'], edgecolor='black')
                ax1.set_title('Abstract Length Distribution', fontweight='bold')
                ax1.set_xlabel('Number of Words')
                ax1.set_ylabel('Frequency')

                # Add mean line
                mean_length = np.mean(abstract_lengths)
                ax1.axvline(mean_length, color='red', linestyle='--',
                          label=f'Mean: {mean_length:.0f} words')
                ax1.legend()

            # 2. Publication years
            years = []
            for paper in papers:
                pub_date = getattr(paper, 'publication_date', '') or ''
                if pub_date:
                    # Extract year from date string
                    year_match = re.search(r'20\d{2}', pub_date)
                    if year_match:
                        years.append(int(year_match.group()))

            if years:
                year_counts = Counter(years)
                sorted_years = sorted(year_counts.keys())
                counts = [year_counts[year] for year in sorted_years]

                bars = ax2.bar(sorted_years, counts, alpha=0.7, color=self.colors['secondary'])
                ax2.set_title('Publications by Year', fontweight='bold')
                ax2.set_xlabel('Year')
                ax2.set_ylabel('Number of Papers')

                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom')

            # 3. Top journals
            journals = []
            for paper in papers:
                journal = getattr(paper, 'journal', '') or 'Unknown'
                if journal and journal != 'Unknown':
                    journals.append(journal)

            if journals:
                journal_counts = Counter(journals)
                top_journals = journal_counts.most_common(10)

                if top_journals:
                    journal_names = [j[0][:30] + '...' if len(j[0]) > 30 else j[0] for j in top_journals]
                    journal_values = [j[1] for j in top_journals]

                    y_pos = np.arange(len(journal_names))
                    bars = ax3.barh(y_pos, journal_values, alpha=0.7, color=self.colors['accent'])
                    ax3.set_yticks(y_pos)
                    ax3.set_yticklabels(journal_names, fontsize=9)
                    ax3.set_xlabel('Number of Papers')
                    ax3.set_title('Top Journals', fontweight='bold')
                    ax3.invert_yaxis()

            # 4. Abstract complexity (readability)
            if abstract_lengths:
                readability_scores = []
                for paper in papers:
                    abstract = getattr(paper, 'abstract', '') or ''
                    if abstract and len(abstract) > 50:
                        try:
                            score = flesch_reading_ease(abstract)
                            readability_scores.append(score)
                        except:
                            pass

                if readability_scores:
                    ax4.hist(readability_scores, bins=15, alpha=0.7,
                            color=self.colors['info'], edgecolor='black')
                    ax4.set_title('Abstract Readability Scores', fontweight='bold')
                    ax4.set_xlabel('Flesch Reading Ease Score')
                    ax4.set_ylabel('Frequency')

                    # Add readability categories
                    ax4.axvline(90, color='green', linestyle=':', alpha=0.7, label='Very Easy')
                    ax4.axvline(60, color='orange', linestyle=':', alpha=0.7, label='Standard')
                    ax4.axvline(30, color='red', linestyle=':', alpha=0.7, label='Difficult')
                    ax4.legend(fontsize=8)

            plt.tight_layout()
            filename = os.path.join(self.output_dir, f"abstract_analysis_{timestamp}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Created abstract analysis dashboard: {filename}")
            return filename

        except Exception as e:
            logger.error(f"❌ Abstract analysis dashboard creation failed: {e}")
            return None

    def _create_text_analytics_dashboard(self, papers: List, query: str) -> Optional[str]:
        """Create text analytics focused on abstracts"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Abstract Text Analytics: {query}', fontsize=16, fontweight='bold')

            # Collect all abstracts
            all_abstracts = []
            for paper in papers:
                abstract = getattr(paper, 'abstract', '') or ''
                if abstract:
                    all_abstracts.append(abstract)

            if not all_abstracts:
                ax1.text(0.5, 0.5, 'No abstracts available', ha='center', va='center',
                        transform=ax1.transAxes, fontsize=14)
                return None

            # 1. Common words (excluding stopwords)
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
                        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                        'this', 'that', 'these', 'those', 'we', 'our', 'study', 'studies',
                        'research', 'method', 'methods', 'analysis', 'using', 'used'}

            all_words = []
            for abstract in all_abstracts:
                words = re.findall(r'\b[a-zA-Z]{3,}\b', abstract.lower())
                words = [w for w in words if w not in stopwords]
                all_words.extend(words)

            if all_words:
                word_counts = Counter(all_words)
                top_words = word_counts.most_common(15)

                words = [w[0] for w in top_words]
                counts = [w[1] for w in top_words]

                y_pos = np.arange(len(words))
                bars = ax1.barh(y_pos, counts, alpha=0.7, color=self.colors['primary'])
                ax1.set_yticks(y_pos)
                ax1.set_yticklabels(words)
                ax1.set_xlabel('Frequency')
                ax1.set_title('Most Common Terms', fontweight='bold')
                ax1.invert_yaxis()

            # 2. Sentence complexity
            sentence_lengths = []
            for abstract in all_abstracts:
                sentences = re.split(r'[.!?]+', abstract)
                for sentence in sentences:
                    words = sentence.split()
                    if len(words) > 3:  # Filter very short sentences
                        sentence_lengths.append(len(words))

            if sentence_lengths:
                ax2.hist(sentence_lengths, bins=20, alpha=0.7,
                        color=self.colors['secondary'], edgecolor='black')
                ax2.set_title('Sentence Length Distribution', fontweight='bold')
                ax2.set_xlabel('Words per Sentence')
                ax2.set_ylabel('Frequency')

                mean_length = np.mean(sentence_lengths)
                ax2.axvline(mean_length, color='red', linestyle='--',
                          label=f'Mean: {mean_length:.1f} words')
                ax2.legend()

            # 3. Abstract word clouds
            if all_abstracts:
                combined_text = ' '.join(all_abstracts)

                # Clean and filter text
                words_for_cloud = [w for w in all_words if len(w) > 3]
                cloud_text = ' '.join(words_for_cloud)

                if cloud_text:
                    wordcloud = WordCloud(
                        width=400, height=300,
                        background_color='white',
                        colormap='viridis',
                        max_words=50,
                        relative_scaling=0.5,
                        random_state=42
                    ).generate(cloud_text)

                    ax3.imshow(wordcloud, interpolation='bilinear')
                    ax3.axis('off')
                    ax3.set_title('Key Terms Word Cloud', fontweight='bold')

            # 4. Abstract statistics
            if all_abstracts:
                stats_data = {
                    'Total Papers': len(papers),
                    'With Abstracts': len(all_abstracts),
                    'Avg Words/Abstract': np.mean([len(abs.split()) for abs in all_abstracts]),
                    'Total Unique Terms': len(set(all_words)),
                    'Avg Readability Score': 0
                }

                # Calculate average readability
                readability_scores = []
                for abstract in all_abstracts:
                    try:
                        score = flesch_reading_ease(abstract)
                        readability_scores.append(score)
                    except:
                        pass

                if readability_scores:
                    stats_data['Avg Readability Score'] = np.mean(readability_scores)

                # Create text display of statistics
                stats_text = '\n'.join([f'{k}: {v:.1f}' if isinstance(v, float) else f'{k}: {v}'
                                      for k, v in stats_data.items()])

                ax4.text(0.1, 0.9, 'Abstract Statistics:', fontsize=14, fontweight='bold',
                        transform=ax4.transAxes, verticalalignment='top')
                ax4.text(0.1, 0.7, stats_text, fontsize=12, transform=ax4.transAxes,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
                ax4.set_xlim(0, 1)
                ax4.set_ylim(0, 1)
                ax4.axis('off')

            plt.tight_layout()
            filename = os.path.join(self.output_dir, f"text_analytics_{timestamp}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Created text analytics dashboard: {filename}")
            return filename

        except Exception as e:
            logger.error(f"❌ Text analytics dashboard creation failed: {e}")
            return None

    def _create_journal_year_analysis(self, papers: List, query: str) -> Optional[str]:
        """Create detailed journal and temporal analysis"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Journal & Temporal Analysis: {query}', fontsize=16, fontweight='bold')

            # Collect data
            journals = []
            years = []
            journal_year_pairs = []

            for paper in papers:
                journal = getattr(paper, 'journal', '') or 'Unknown'
                pub_date = getattr(paper, 'publication_date', '') or ''

                year = None
                if pub_date:
                    year_match = re.search(r'20\d{2}', pub_date)
                    if year_match:
                        year = int(year_match.group())

                if journal != 'Unknown':
                    journals.append(journal)
                if year:
                    years.append(year)
                if journal != 'Unknown' and year:
                    journal_year_pairs.append((journal, year))

            # 1. Journal impact (by count)
            if journals:
                journal_counts = Counter(journals)
                top_journals = journal_counts.most_common(10)

                colors = plt.cm.Set3(np.linspace(0, 1, len(top_journals)))
                sizes = [j[1] for j in top_journals]
                labels = [j[0][:25] + '...' if len(j[0]) > 25 else j[0] for j in top_journals]

                wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                                  colors=colors, startangle=90)
                ax1.set_title('Journal Distribution', fontweight='bold')

                # Improve label readability
                for text in texts:
                    text.set_fontsize(8)

            # 2. Publications over time
            if years:
                year_counts = Counter(years)
                sorted_years = sorted(year_counts.keys())
                counts = [year_counts[year] for year in sorted_years]

                ax2.plot(sorted_years, counts, marker='o', linewidth=2,
                        markersize=6, color=self.colors['primary'])
                ax2.fill_between(sorted_years, counts, alpha=0.3, color=self.colors['primary'])
                ax2.set_title('Publication Trends Over Time', fontweight='bold')
                ax2.set_xlabel('Year')
                ax2.set_ylabel('Number of Publications')
                ax2.grid(True, alpha=0.3)

                # Add value labels
                for x, y in zip(sorted_years, counts):
                    ax2.annotate(f'{y}', (x, y), textcoords="offset points",
                               xytext=(0,10), ha='center', fontsize=9)

            # 3. Journal-Year heatmap
            if journal_year_pairs and len(set(journals)) > 1 and len(set(years)) > 1:
                # Create heatmap data
                journal_list = list(set(journals))[:10]  # Top 10 journals
                year_list = sorted(list(set(years)))

                heatmap_data = np.zeros((len(journal_list), len(year_list)))

                for journal, year in journal_year_pairs:
                    if journal in journal_list:
                        j_idx = journal_list.index(journal)
                        y_idx = year_list.index(year)
                        heatmap_data[j_idx, y_idx] += 1

                im = ax3.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
                ax3.set_xticks(range(len(year_list)))
                ax3.set_xticklabels(year_list)
                ax3.set_yticks(range(len(journal_list)))
                ax3.set_yticklabels([j[:20] + '...' if len(j) > 20 else j for j in journal_list])
                ax3.set_title('Journal-Year Publication Matrix', fontweight='bold')
                ax3.set_xlabel('Year')

                # Add colorbar
                plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

                # Add text annotations for non-zero values
                for i in range(len(journal_list)):
                    for j in range(len(year_list)):
                        if heatmap_data[i, j] > 0:
                            ax3.text(j, i, int(heatmap_data[i, j]),
                                   ha="center", va="center", color="black", fontweight='bold')

            # 4. Summary statistics
            summary_stats = {
                'Total Papers': len(papers),
                'Unique Journals': len(set(journals)) if journals else 0,
                'Year Range': f"{min(years)}-{max(years)}" if years else 'N/A',
                'Most Prolific Journal': max(Counter(journals), key=Counter(journals).get) if journals else 'N/A',
                'Most Active Year': max(Counter(years), key=Counter(years).get) if years else 'N/A'
            }

            # Display summary
            y_pos = 0.9
            ax4.text(0.1, y_pos, 'Dataset Summary:', fontsize=14, fontweight='bold',
                    transform=ax4.transAxes)

            y_pos -= 0.15
            for key, value in summary_stats.items():
                ax4.text(0.1, y_pos, f'{key}:', fontsize=12, fontweight='bold',
                        transform=ax4.transAxes)
                ax4.text(0.5, y_pos, str(value), fontsize=12, transform=ax4.transAxes)
                y_pos -= 0.1

            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')

            plt.tight_layout()
            filename = os.path.join(self.output_dir, f"journal_year_analysis_{timestamp}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Created journal-year analysis: {filename}")
            return filename

        except Exception as e:
            logger.error(f"❌ Journal-year analysis creation failed: {e}")
            return None

    def _create_keyword_analysis(self, papers: List, query: str) -> Optional[str]:
        """Create keyword and term analysis"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Keyword & Term Analysis: {query}', fontsize=16, fontweight='bold')

            # Collect abstracts and titles
            abstracts = []
            titles = []
            for paper in papers:
                abstract = getattr(paper, 'abstract', '') or ''
                title = getattr(paper, 'title', '') or ''
                if abstract:
                    abstracts.append(abstract)
                if title:
                    titles.append(title)

            # 1. Title word analysis
            if titles:
                title_words = []
                stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}

                for title in titles:
                    words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
                    words = [w for w in words if w not in stopwords]
                    title_words.extend(words)

                if title_words:
                    title_counts = Counter(title_words)
                    top_title_words = title_counts.most_common(12)

                    words = [w[0] for w in top_title_words]
                    counts = [w[1] for w in top_title_words]

                    bars = ax1.bar(range(len(words)), counts, alpha=0.7, color=self.colors['accent'])
                    ax1.set_xticks(range(len(words)))
                    ax1.set_xticklabels(words, rotation=45, ha='right')
                    ax1.set_title('Most Common Title Terms', fontweight='bold')
                    ax1.set_ylabel('Frequency')

                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom')

            # 2. Abstract key phrases (bigrams)
            if abstracts:
                all_text = ' '.join(abstracts).lower()
                words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text)

                # Create bigrams
                bigrams = []
                for i in range(len(words) - 1):
                    if words[i] not in {'this', 'that', 'with', 'from', 'they', 'were', 'have', 'been'}:
                        bigram = f"{words[i]} {words[i+1]}"
                        bigrams.append(bigram)

                if bigrams:
                    bigram_counts = Counter(bigrams)
                    top_bigrams = bigram_counts.most_common(10)

                    phrases = [b[0] for b in top_bigrams]
                    counts = [b[1] for b in top_bigrams]

                    y_pos = np.arange(len(phrases))
                    bars = ax2.barh(y_pos, counts, alpha=0.7, color=self.colors['info'])
                    ax2.set_yticks(y_pos)
                    ax2.set_yticklabels(phrases, fontsize=10)
                    ax2.set_xlabel('Frequency')
                    ax2.set_title('Common Phrases (Bigrams)', fontweight='bold')
                    ax2.invert_yaxis()

            # 3. Query term analysis
            if query and abstracts:
                query_terms = query.lower().split()
                term_frequencies = {}

                for term in query_terms:
                    if len(term) > 2:
                        count = sum(1 for abstract in abstracts if term in abstract.lower())
                        if count > 0:
                            term_frequencies[term] = count

                if term_frequencies:
                    terms = list(term_frequencies.keys())
                    freqs = list(term_frequencies.values())

                    bars = ax3.bar(terms, freqs, alpha=0.7, color=self.colors['secondary'])
                    ax3.set_title(f'Query Terms in Abstracts', fontweight='bold')
                    ax3.set_ylabel('Papers Containing Term')
                    ax3.set_xlabel('Query Terms')

                    # Add percentage labels
                    total_papers = len(abstracts)
                    for bar, freq in zip(bars, freqs):
                        height = bar.get_height()
                        percentage = (freq / total_papers) * 100
                        ax3.text(bar.get_x() + bar.get_width()/2., height,
                               f'{percentage:.1f}%', ha='center', va='bottom')

            # 4. Term co-occurrence network (simplified)
            if abstracts and len(abstracts) > 3:
                # Get top terms
                all_words = []
                for abstract in abstracts:
                    words = re.findall(r'\b[a-zA-Z]{4,}\b', abstract.lower())
                    words = [w for w in words if w not in {'this', 'that', 'with', 'from', 'they', 'were', 'have', 'been', 'using', 'used', 'study', 'method', 'analysis'}]
                    all_words.extend(words)

                word_counts = Counter(all_words)
                top_terms = [w[0] for w in word_counts.most_common(8)]

                # Create co-occurrence matrix
                G = nx.Graph()
                for abstract in abstracts:
                    abstract_terms = [w for w in top_terms if w in abstract.lower()]
                    # Add edges between co-occurring terms
                    for i, term1 in enumerate(abstract_terms):
                        for term2 in abstract_terms[i+1:]:
                            if G.has_edge(term1, term2):
                                G[term1][term2]['weight'] += 1
                            else:
                                G.add_edge(term1, term2, weight=1)

                if G.nodes():
                    pos = nx.spring_layout(G, k=1, iterations=50)

                    # Draw network
                    nx.draw_networkx_nodes(G, pos, node_color=self.colors['primary'],
                                         node_size=300, alpha=0.8, ax=ax4)
                    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax4)

                    # Draw edges with varying thickness
                    edges = G.edges()
                    weights = [G[u][v]['weight'] for u, v in edges]
                    if weights:
                        max_weight = max(weights)
                        for (u, v), weight in zip(edges, weights):
                            width = 1 + 3 * (weight / max_weight)
                            nx.draw_networkx_edges(G, pos, [(u, v)], width=width,
                                                 alpha=0.6, ax=ax4)

                    ax4.set_title('Term Co-occurrence Network', fontweight='bold')
                    ax4.axis('off')

            plt.tight_layout()
            filename = os.path.join(self.output_dir, f"keyword_analysis_{timestamp}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Created keyword analysis: {filename}")
            return filename

        except Exception as e:
            logger.error(f"❌ Keyword analysis creation failed: {e}")
            return None

    def _create_abstract_similarity_network(self, papers: List, query: str) -> Optional[str]:
        """Create paper similarity network based on abstracts"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if len(papers) < 3:
                logger.warning("Need at least 3 papers for similarity network")
                return None

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(f'Abstract Similarity Network: {query}', fontsize=16, fontweight='bold')

            # Collect abstracts with paper info
            paper_data = []
            for i, paper in enumerate(papers):
                abstract = getattr(paper, 'abstract', '') or ''
                title = getattr(paper, 'title', '') or f'Paper {i+1}'
                if abstract:
                    paper_data.append({
                        'id': i,
                        'title': title[:30] + '...' if len(title) > 30 else title,
                        'abstract': abstract,
                        'journal': getattr(paper, 'journal', '') or 'Unknown'
                    })

            if len(paper_data) < 3:
                logger.warning("Need at least 3 papers with abstracts")
                return None

            # Simple similarity based on word overlap
            def jaccard_similarity(text1, text2):
                words1 = set(re.findall(r'\b[a-zA-Z]{4,}\b', text1.lower()))
                words2 = set(re.findall(r'\b[a-zA-Z]{4,}\b', text2.lower()))
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                return len(intersection) / len(union) if len(union) > 0 else 0

            # Create similarity network
            G = nx.Graph()
            similarity_threshold = 0.05  # Adjust based on data

            for i, paper1 in enumerate(paper_data):
                G.add_node(i, title=paper1['title'], journal=paper1['journal'])

            for i, paper1 in enumerate(paper_data):
                for j, paper2 in enumerate(paper_data[i+1:], i+1):
                    similarity = jaccard_similarity(paper1['abstract'], paper2['abstract'])
                    if similarity > similarity_threshold:
                        G.add_edge(i, j, weight=similarity)

            # Layout and draw network
            if G.nodes():
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

                # Node colors by journal
                journals = list(set([paper['journal'] for paper in paper_data]))
                journal_colors = dict(zip(journals, plt.cm.Set3(np.linspace(0, 1, len(journals)))))
                node_colors = [journal_colors[paper_data[node]['journal']] for node in G.nodes()]

                # Draw network
                nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500,
                                     alpha=0.8, ax=ax1)

                # Draw edges with varying thickness
                if G.edges():
                    edges = G.edges()
                    weights = [G[u][v]['weight'] for u, v in edges]
                    max_weight = max(weights) if weights else 1

                    for (u, v), weight in zip(edges, weights):
                        width = 1 + 4 * (weight / max_weight)
                        nx.draw_networkx_edges(G, pos, [(u, v)], width=width,
                                             alpha=0.6, edge_color='gray', ax=ax1)

                # Add labels
                labels = {i: f"P{i+1}" for i in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax1)

                ax1.set_title('Paper Similarity Network', fontweight='bold')
                ax1.axis('off')

                # Create legend for journals
                handles = [plt.Rectangle((0,0),1,1, color=journal_colors[journal], alpha=0.8)
                          for journal in journals]
                legend_labels = [j[:20] + '...' if len(j) > 20 else j for j in journals]
                ax1.legend(handles, legend_labels, loc='upper left', bbox_to_anchor=(0, 1))

            # Similarity matrix heatmap
            n_papers = len(paper_data)
            similarity_matrix = np.zeros((n_papers, n_papers))

            for i, paper1 in enumerate(paper_data):
                for j, paper2 in enumerate(paper_data):
                    if i != j:
                        similarity_matrix[i, j] = jaccard_similarity(paper1['abstract'], paper2['abstract'])

            im = ax2.imshow(similarity_matrix, cmap='YlOrRd', aspect='auto')
            ax2.set_xticks(range(n_papers))
            ax2.set_yticks(range(n_papers))
            ax2.set_xticklabels([f'P{i+1}' for i in range(n_papers)])
            ax2.set_yticklabels([f'P{i+1}' for i in range(n_papers)])
            ax2.set_title('Paper Similarity Matrix', fontweight='bold')

            # Add colorbar
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label='Jaccard Similarity')

            # Add text annotations for high similarity values
            for i in range(n_papers):
                for j in range(n_papers):
                    if similarity_matrix[i, j] > 0.1:  # Show values above threshold
                        ax2.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold', fontsize=8)

            plt.tight_layout()
            filename = os.path.join(self.output_dir, f"similarity_network_{timestamp}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Created similarity network: {filename}")
            return filename

        except Exception as e:
            logger.error(f"❌ Similarity network creation failed: {e}")
            return None

    def _create_comprehensive_summary(self, papers: List, query: str) -> Optional[str]:
        """Create comprehensive summary visualization"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            fig.suptitle(f'Comprehensive Summary: {query}', fontsize=16, fontweight='bold')

            # Collect data
            abstracts = [getattr(p, 'abstract', '') or '' for p in papers if getattr(p, 'abstract', '')]
            years = []
            journals = []

            for paper in papers:
                pub_date = getattr(paper, 'publication_date', '') or ''
                if pub_date:
                    year_match = re.search(r'20\d{2}', pub_date)
                    if year_match:
                        years.append(int(year_match.group()))

                journal = getattr(paper, 'journal', '') or ''
                if journal:
                    journals.append(journal)

            # 1. Dataset overview (top-left, large)
            ax1 = fig.add_subplot(gs[0, :2])

            overview_stats = [
                f"Total Papers: {len(papers)}",
                f"With Abstracts: {len(abstracts)}",
                f"Unique Journals: {len(set(journals))}",
                f"Year Range: {min(years)}-{max(years)}" if years else "Year Range: N/A",
                f"Query: {query}",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            ]

            y_pos = 0.9
            for stat in overview_stats:
                ax1.text(0.05, y_pos, stat, fontsize=14, transform=ax1.transAxes,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
                y_pos -= 0.13

            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.set_title('Dataset Overview', fontweight='bold', fontsize=14)
            ax1.axis('off')

            # 2. Quick metrics (top-right)
            ax2 = fig.add_subplot(gs[0, 2])

            if abstracts:
                avg_abstract_length = np.mean([len(abs.split()) for abs in abstracts])
                total_words = sum([len(abs.split()) for abs in abstracts])

                readability_scores = []
                for abstract in abstracts:
                    try:
                        score = flesch_reading_ease(abstract)
                        readability_scores.append(score)
                    except:
                        pass

                avg_readability = np.mean(readability_scores) if readability_scores else 0

                metrics = [
                    f"Avg Abstract Length:\n{avg_abstract_length:.0f} words",
                    f"Total Words:\n{total_words:,}",
                    f"Avg Readability:\n{avg_readability:.1f}"
                ]

                y_positions = [0.8, 0.5, 0.2]
                for metric, y in zip(metrics, y_positions):
                    ax2.text(0.5, y, metric, fontsize=11, ha='center', va='center',
                            transform=ax2.transAxes,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.set_title('Key Metrics', fontweight='bold')
            ax2.axis('off')

            # 3. Timeline (middle-left)
            ax3 = fig.add_subplot(gs[1, 0])

            if years:
                year_counts = Counter(years)
                sorted_years = sorted(year_counts.keys())
                counts = [year_counts[year] for year in sorted_years]

                ax3.plot(sorted_years, counts, marker='o', linewidth=2, markersize=4)
                ax3.fill_between(sorted_years, counts, alpha=0.3)
                ax3.set_title('Publication Timeline', fontweight='bold', fontsize=12)
                ax3.set_ylabel('Papers')
                ax3.tick_params(axis='x', labelsize=8)
                ax3.tick_params(axis='y', labelsize=8)

            # 4. Top journals (middle-center)
            ax4 = fig.add_subplot(gs[1, 1])

            if journals:
                journal_counts = Counter(journals)
                top_journals = journal_counts.most_common(5)

                labels = [j[0][:15] + '...' if len(j[0]) > 15 else j[0] for j in top_journals]
                sizes = [j[1] for j in top_journals]

                colors = plt.cm.Set3(np.linspace(0, 1, len(top_journals)))
                ax4.pie(sizes, labels=labels, autopct='%1.0f%%', colors=colors, startangle=90)
                ax4.set_title('Top Journals', fontweight='bold', fontsize=12)

            # 5. Word cloud (middle-right)
            ax5 = fig.add_subplot(gs[1, 2])

            if abstracts:
                all_text = ' '.join(abstracts).lower()
                words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text)
                stopwords = {'this', 'that', 'with', 'from', 'they', 'were', 'have', 'been',
                            'using', 'used', 'study', 'method', 'analysis', 'research'}
                words = [w for w in words if w not in stopwords]

                if words:
                    word_text = ' '.join(words)
                    wordcloud = WordCloud(width=300, height=200, background_color='white',
                                        colormap='viridis', max_words=30, relative_scaling=0.5,
                                        random_state=42).generate(word_text)

                    ax5.imshow(wordcloud, interpolation='bilinear')
                    ax5.axis('off')
                    ax5.set_title('Key Terms', fontweight='bold', fontsize=12)

            # 6. Bottom row: Abstract length distribution
            ax6 = fig.add_subplot(gs[2, :])

            if abstracts:
                lengths = [len(abs.split()) for abs in abstracts]
                ax6.hist(lengths, bins=min(15, len(set(lengths))), alpha=0.7,
                        color=self.colors['primary'], edgecolor='black')
                ax6.set_title('Abstract Length Distribution', fontweight='bold', fontsize=12)
                ax6.set_xlabel('Words per Abstract')
                ax6.set_ylabel('Frequency')

                mean_length = np.mean(lengths)
                ax6.axvline(mean_length, color='red', linestyle='--',
                          label=f'Mean: {mean_length:.0f} words')
                ax6.legend()

            plt.tight_layout()
            filename = os.path.join(self.output_dir, f"comprehensive_summary_{timestamp}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Created comprehensive summary: {filename}")
            return filename

        except Exception as e:
            logger.error(f"❌ Comprehensive summary creation failed: {e}")
            return None