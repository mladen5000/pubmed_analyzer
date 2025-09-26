# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **PubMed Analyzer** - a modular scientific literature analysis pipeline that combines full-text PDF processing, Retrieval-Augmented Generation (RAG), vector search, and advanced analytics for biomedical research. The project features a clean modular architecture with separate components for search, PDF fetching, analysis, and utilities.

## Core Architecture & Entry Points

The project has evolved into multiple specialized entry points:

### 1. Main CLI Interface (`pubmed_analyzer.py`)
**Primary entry point** with two optimized modes:
- **ABSTRACTS MODE**: Ultra-fast abstract-only analysis (recommended for exploration)
- **FULL MODE**: Comprehensive analysis with robust PDF downloading

### 2. Enhanced Pipeline (`enhanced_main.py`)
Advanced NLP/ML analysis with enhanced RAG capabilities, topic modeling, and comprehensive visualizations.

### 3. Legacy Scripts
- `main_new.py`: Original modular pipeline (moved to legacy)
- `fetch_pubmed.py`: Simple fetching utilities

## Core Features

1. **Paper Search & Retrieval**: PubMed searches using NCBI E-utilities API
2. **Multi-Strategy PDF Downloading**: Robust error handling with multiple fallback strategies
3. **Vector Search**: FAISS indices for semantic search using sentence transformers
4. **RAG Integration**: OpenAI and DeepSeek API support for question-answering
5. **Advanced Analytics**: Topic modeling, sentiment analysis, network analysis, visualizations
6. **Abstract-Only Analysis**: Fast processing without PDF requirements

### Modular Components

- **`pubmed_analyzer/core/`**: Core functionality
  - `search.py`: PubMedSearcher for paper discovery
  - `id_converter.py`: PMID to PMC ID conversion
  - `pdf_fetcher.py`: Legacy PDF fetching
  - `robust_pdf_fetcher.py`: Enhanced PDF downloading with retry logic
  - `nlp_analyzer.py`: Advanced NLP analysis (topic modeling, sentiment)
  - `rag_analyzer.py`: RAG and LLM integration
  - `llm_analyzer.py`: LLM-powered analysis
  - `markdown_converter.py`: PDF to Markdown conversion
- **`pubmed_analyzer/api/`**: API interfaces
  - `pdf_fetcher_api.py`: Unified PDF fetching API
- **`pubmed_analyzer/models/`**: Data models
  - `paper.py`: Paper data structure and metadata handling
- **`pubmed_analyzer/utils/`**: Utilities
  - `ncbi_client.py`: NCBI API client with rate limiting
  - `validators.py`: PDF and data validation
  - `visualizer.py`: Enhanced visualizations
  - `abstract_visualizer.py`: Abstract-optimized visualizations

### Directory Structure

- `pdfs/`: Downloaded PDF files
- `sections/`: Extracted paper sections as JSON
- `vector_indices/`: FAISS indices and metadata
- `markdown/`: PDF to Markdown conversions

## Setup Requirements

### 1. Required Configuration
**CRITICAL**: Set your email address for NCBI API access:
```bash
# Environment variable (recommended)
export PUBMED_EMAIL="your.email@example.com"

# Or use command line argument
--email your.email@example.com
```
NCBI requires a valid email address for API access.

### 2. Optional but Recommended
- **NCBI API Key**: Get from [NCBI Account Settings](https://www.ncbi.nlm.nih.gov/account/settings/) for higher rate limits
- **Environment Variables**:
  ```bash
  export NCBI_API_KEY="your_key"       # For higher NCBI rate limits
  export OPENAI_API_KEY="your_key"     # For LLM integration
  export DEEPSEEK_API_KEY="your_key"   # For cheaper LLM alternative
  ```

### 3. Installation and Run Commands

#### Main CLI Interface (Recommended)
```bash
# Install dependencies
uv sync

# ABSTRACTS MODE: Ultra-fast analysis (recommended for exploration)
uv run python pubmed_analyzer.py abstracts --query "machine learning" --max-papers 100 --visualizations

# FULL MODE: Comprehensive analysis with PDF downloads
uv run python pubmed_analyzer.py full --query "COVID-19" --max-papers 50 --pdf-dir covid_pdfs

# Get help
uv run python pubmed_analyzer.py --help
uv run python pubmed_analyzer.py abstracts --help
uv run python pubmed_analyzer.py full --help
```

#### Enhanced Pipeline (Advanced Features)
```bash
# Run enhanced analysis with advanced NLP/ML
uv run python enhanced_main.py --query "CRISPR gene editing" --max-papers 50
```

#### Legacy Interface
```bash
# Original modular pipeline (still functional)
uv run python main_new.py --query "your search terms" --max-papers 50
```

### 4. Enhanced PDF Download System

The system features a **multi-source PDF fetching architecture** with significant success rate improvements:

#### **Enhanced Mode (Default - Recommended)**
- **Success Rate**: 60-80% (3x improvement over standard)
- **8-Tier Strategy System**:
  - **Tier 0**: EuropePMC (highest success rate for PMC papers)
  - **Tier 1**: PMC OA Service (official NCBI service)
  - **Tier 2**: Direct PMC & DOI Redirect (publisher access)
  - **Tier 3**: arXiv API (official API for arXiv papers)
  - **Tier 4**: paperscraper (arXiv, bioRxiv, medRxiv preprints)
  - **Tier 5**: PyPaperBot (broader access - educational use only)

#### **Standard Mode (Official Sources Only)**
- **Success Rate**: 20-40% (PMC Open Access only)
- Uses only official NCBI and publisher sources
- More conservative approach for institutional use

#### **Usage Examples:**
```bash
# Enhanced mode (default) - higher success rates
uv run python pubmed_analyzer.py full --query "COVID-19" --max-papers 50

# Standard mode - official sources only
uv run python pubmed_analyzer.py full --query "COVID-19" --max-papers 50 --no-enhanced

# Enhanced pipeline with third-party sources
uv run python enhanced_main.py --query "CRISPR gene editing" --max-papers 50
```

#### **Technical Features:**
- **Circuit breakers**: Temporarily disable failing sources
- **Rate limiting**: Respects API limits for all sources
- **Smart fallbacks**: Tries multiple sources per paper
- **Validation**: Checks PDF content and headers
- **Exponential backoff**: Progressive retry delays

#### **Legal Considerations:**
- **Fully Legal**: arXiv API, paperscraper (uses official APIs)
- **Educational Use**: PyPaperBot (marked for educational purposes)
- **Publisher Compliance**: Respects robots.txt and rate limits
- **Configurable**: Can disable specific sources if needed

#### **Known Limitations:**
- **PMC Coverage**: Only ~30% of papers have PMC IDs
- **Access Restrictions**: Many papers still require subscriptions
- **Regional Blocks**: Some publishers block automated access
- **Preprint Focus**: Higher success with arXiv/bioRxiv papers

#### **Troubleshooting:**
- Use ABSTRACTS mode for instant analysis without PDFs
- Enhanced mode automatically enabled - disable with `--no-enhanced`
- Check strategy performance in logs for optimization
- Consider institutional VPN for better publisher access

The enhanced system maintains full backward compatibility while dramatically improving PDF acquisition success rates.

## Key Dependencies

- **PDF Processing**: PyMuPDF (`fitz`)
- **Enhanced PDF Fetching**: paperscraper, PyPaperBot, arxiv (for multi-source access)
- **NLP**: NLTK, spaCy, TextBlob, sentence-transformers
- **ML**: scikit-learn, FAISS for vector search
- **Visualization**: matplotlib, seaborn, wordcloud
- **APIs**: requests for NCBI, openai/ollama for LLM integration
- **Data**: pandas, numpy

## Output Files

The application generates:
- `enhanced_analysis_report.md`: Comprehensive analysis report
- `enhanced_analysis_dashboard.png`: Visualization dashboard
- `comprehensive_analysis.json`: All analysis results in JSON
- `rag_example.json`: Example RAG query results (if LLM APIs configured)

## Important Notes

- Requires internet connection for PubMed API and PDF downloads
- SpaCy model `en_core_web_sm` is auto-downloaded if not present
- FAISS indices are persistent and stored in `vector_indices/`
- The application is designed for single execution runs, not as a service