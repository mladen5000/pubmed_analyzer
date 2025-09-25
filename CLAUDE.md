# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **PubMed Analyzer** - a modular scientific literature analysis pipeline that combines full-text PDF processing, Retrieval-Augmented Generation (RAG), vector search, and advanced analytics for biomedical research. The project features a clean modular architecture with separate components for search, PDF fetching, analysis, and utilities.

## Core Architecture

The application is built around the `ModularPubMedPipeline` class in `main_new.py` which orchestrates:

1. **Paper Search & Retrieval**: PubMed searches using NCBI E-utilities API
2. **Full-text Processing**: Multi-strategy PDF downloads with robust error handling
3. **Vector Search**: FAISS indices for semantic search using sentence transformers
4. **RAG Integration**: OpenAI and DeepSeek API support for question-answering
5. **Advanced Analytics**: Topic modeling, sentiment analysis, network analysis, visualizations

### Modular Components

- **`pubmed_analyzer/core/`**: Core functionality
  - `search.py`: PubMedSearcher for paper discovery
  - `id_converter.py`: PMID to PMC ID conversion
  - `pdf_fetcher.py`: Multi-strategy PDF downloading with retry logic
- **`pubmed_analyzer/models/`**: Data models
  - `paper.py`: Paper data structure and metadata handling
- **`pubmed_analyzer/utils/`**: Utilities
  - `ncbi_client.py`: NCBI API client with rate limiting
  - `validators.py`: PDF and data validation

### Directory Structure

- `pdfs/`: Downloaded PDF files
- `sections/`: Extracted paper sections as JSON
- `vector_indices/`: FAISS indices and metadata
- `markdown/`: PDF to Markdown conversions

## Setup Requirements

### 1. Required Configuration
**CRITICAL**: You must edit the email address when running the pipeline:
```python
# In main_new.py or when initializing ModularPubMedPipeline
EMAIL = "your.email@example.com"  # CHANGE THIS TO YOUR REAL EMAIL
```
NCBI requires a valid email address for API access.

### 2. Optional but Recommended
- **NCBI API Key**: Get from [NCBI Account Settings](https://www.ncbi.nlm.nih.gov/account/settings/) for higher rate limits
- **Environment Variables**:
  ```bash
  export OPENAI_API_KEY="your_key"     # For LLM integration
  export DEEPSEEK_API_KEY="your_key"   # For cheaper LLM alternative
  ```

### 3. Installation and Run Commands
```bash
# Install dependencies
uv sync

# Run the modular application
uv run python main_new.py --query "your search terms" --max-papers 50

# Alternative if virtual environment is activated
python main_new.py --help

# Available command-line arguments:
# --query: PubMed search query
# --max-papers: Maximum number of papers (default: 100)
# --email: Email for NCBI API
# --api-key: NCBI API key (optional)
```

### 4. PDF Download Strategy
The system uses a robust multi-strategy approach:
- **Multiple URL patterns**: Tries 3 different URL formats per paper
- **Exponential backoff**: 3 retry attempts with progressive delays
- **Success rate enforcement**: Requires 30% minimum success rate
- **Batch processing**: Downloads in batches of 5 with rate limiting
- **Validation**: Checks file size and PDF headers

Even with this robust system, some PDFs may fail due to:
- Institutional access requirements
- Publisher blocking of automated downloads
- Papers not being truly open access

The application continues analysis using abstracts when PDFs aren't available.

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