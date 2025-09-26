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

### 4. PDF Download Strategy & Known Issues
The system uses a robust multi-strategy approach:
- **Multiple URL patterns**: Tries different URL formats per paper
- **Exponential backoff**: Multiple retry attempts with progressive delays
- **Success rate enforcement**: Configurable minimum success rate thresholds
- **Batch processing**: Downloads in batches with rate limiting
- **Validation**: Checks file size and PDF headers

**Known Issues:**
- **PMC ID Conversion**: Some papers may not have PMC IDs, limiting PDF download options
- **Success Rates**: Expect 20-40% success rate due to access restrictions
- **Institutional Access**: Many papers require subscription access
- **Publisher Blocking**: Some publishers block automated downloads

**Troubleshooting:**
- Use ABSTRACTS mode for fast analysis without PDF requirements
- Check PMID to PMC ID conversion in logs
- Verify NCBI API access with proper email/API key
- Consider institutional access for higher PDF success rates

The application gracefully handles failures and continues analysis using abstracts when PDFs aren't available.

## Key Dependencies

- **PDF Processing**: PyMuPDF (`fitz`)
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