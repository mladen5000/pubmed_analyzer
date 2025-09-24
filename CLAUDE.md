# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **PubMed Analyzer** - a scientific literature analysis pipeline that combines full-text PDF processing, Retrieval-Augmented Generation (RAG), vector search, and advanced analytics for biomedical research. The project is a single-file Python application (`main.py`) that provides comprehensive analysis of PubMed research papers.

## Core Architecture

The application is built around the `EnhancedPubMedAnalyzer` class which provides:

1. **Paper Search & Retrieval**: Searches PubMed using NCBI E-utilities API
2. **Full-text Processing**: Downloads and processes PDFs using PyMuPDF, extracts sections
3. **Vector Search**: Creates FAISS indices for semantic search using sentence transformers
4. **RAG Integration**: Supports OpenAI and DeepSeek APIs for question-answering
5. **Advanced Analytics**: Topic modeling, sentiment analysis, network analysis, visualizations

### Key Components

- **Data Storage**: Papers stored in memory, full-texts in `full_texts` dict, sections in `sections` dict
- **Vector Indices**: FAISS indices for abstracts, full-text chunks, and paper sections
- **Directory Structure**:
  - `pdfs/`: Downloaded PDF files
  - `sections/`: Extracted paper sections as JSON
  - `vector_indices/`: FAISS indices and metadata

## Setup Requirements

### 1. Required Configuration
**CRITICAL**: You must edit `main.py` and change the email address:
```python
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

# Run the application
uv run python main.py

# Alternative if virtual environment is activated
python main.py
```

### 4. PDF Download Issues
Even with proper setup, PDF downloads may fail with 403 errors because:
- Many papers require institutional access
- Publishers block automated downloads
- Some papers aren't truly open access

The application will continue analysis using abstracts when PDFs aren't available.

### Search Parameters
Modify these variables in the `main()` function:
- `SEARCH_QUERY`: PubMed search query
- `MAX_PAPERS`: Maximum number of papers to analyze
- `START_DATE`/`END_DATE`: Date range for search

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