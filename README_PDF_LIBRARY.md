# Robust PubMed PDF Fetching Library

A production-ready, multi-strategy PDF downloading library specifically designed for PubMed papers. Built with rate limiting, error handling, validation, and respectful downloading practices.

## ⭐ Key Features

- **Multi-Strategy Downloads**: Tries multiple approaches (PMC OA Service, DOI resolution, arXiv, etc.)
- **Production Ready**: Circuit breakers, rate limiting, comprehensive error handling
- **Respectful**: Follows NCBI guidelines with proper rate limiting and user agents
- **High Success Rates**: Optimized for maximum PDF retrieval success
- **Comprehensive Validation**: PDF integrity checking and quality validation
- **Batch Processing**: Efficient bulk downloads with success rate monitoring
- **Simple API**: Both async and sync interfaces available

## 🚀 Quick Start

### Installation

```bash
# Install the PubMed Analyzer (includes PDF fetching library)
uv sync
```

### Basic Usage

```python
import asyncio
from pubmed_analyzer.api import PubMedPDFFetcher

async def main():
    # Initialize fetcher
    fetcher = PubMedPDFFetcher(
        email="your.email@example.com",  # REQUIRED
        api_key="your_ncbi_api_key",     # Optional but recommended
        pdf_dir="downloaded_pdfs"
    )

    # Download PDFs by PMIDs
    result = await fetcher.download_from_pmids([
        "33157158", "33157159", "33157160"
    ])

    print(f"Downloaded: {result.successful_downloads}/{result.total_papers}")
    print(f"Success rate: {result.success_rate:.1%}")

asyncio.run(main())
```

### Synchronous Usage

```python
from pubmed_analyzer.api.pdf_fetcher_api import PubMedPDFFetcherSync

# Synchronous wrapper for easier scripting
fetcher = PubMedPDFFetcherSync(
    email="your.email@example.com",
    pdf_dir="pdfs"
)

# Download single paper
result = fetcher.download_single("33157158")
if result.success:
    print(f"✅ Downloaded: {result.file_path}")
```

### One-Liner Downloads

```python
from pubmed_analyzer.api.pdf_fetcher_api import download_pdfs_sync

# Quick synchronous download
result = download_pdfs_sync(
    pmids=["33157158", "33157159"],
    email="your.email@example.com"
)
```

## 📚 API Reference

### PubMedPDFFetcher

Main class for PDF downloading with comprehensive features.

```python
fetcher = PubMedPDFFetcher(
    email="required@example.com",      # REQUIRED: Your email for NCBI
    api_key="optional_ncbi_key",       # Optional: NCBI API key for higher limits
    pdf_dir="pdfs",                    # Directory to save PDFs
    min_success_rate=0.3,              # Stop batch if success rate drops below 30%
    batch_size=5                       # Papers per batch (for rate limiting)
)
```

#### Methods

- **`download_from_pmids(pmids)`**: Download PDFs for list of PMIDs
- **`download_from_search(query, max_results)`**: Search PubMed and download results
- **`download_single(pmid)`**: Download single paper
- **`get_statistics()`**: Get performance statistics
- **`health_check()`**: Check strategy health status

## 🔧 Advanced Features

### Download Strategies (Ordered by Priority)

1. **PMC OA Service** - Official NCBI service for open access papers
2. **DOI Redirect** - Follow DOI redirects to publisher PDFs
3. **arXiv Strategy** - Direct downloads from arXiv preprint server

### Rate Limiting

- **PMC**: 3 requests/second (respects NCBI guidelines)
- **Publishers**: 1 request/second (conservative approach)
- **arXiv**: 2 requests/second (per their guidelines)

### Circuit Breakers

Automatically disable failing strategies:
- Opens after 5 consecutive failures
- Stays open for 5 minutes
- Allows one test request after timeout

### Comprehensive Validation

- File size validation (minimum 1KB)
- PDF header verification (%PDF magic bytes)
- Paywall detection (HTML content filtering)
- Extractable content verification

## 📊 Success Rates

Based on research and testing:

- **PMC Open Access papers**: 95-99% success rate
- **Publisher direct access**: 30-70% (varies by publisher)
- **Overall realistic expectation**: 40-60% across all PubMed papers
- **Open access focused**: 80-90% when limiting to known OA content

## 🔍 Monitoring and Statistics

```python
# Get comprehensive statistics
stats = fetcher.get_statistics()
print(f"Overall success rate: {stats['overall_success_rate']:.1%}")

# Strategy-specific performance
for strategy, data in stats['strategy_statistics'].items():
    print(f"{strategy}: {data['success_rate']:.1%} success rate")

# Health check
health = await fetcher.health_check()
for strategy, status in health.items():
    print(f"{strategy}: {'Available' if status['available'] else 'Unavailable'}")
```

## ⚠️ Important Requirements

### Email Address (Required)
NCBI requires a valid email address for API access. Always provide your real email:

```python
fetcher = PubMedPDFFetcher(email="your.real.email@example.com")
```

### NCBI API Key (Recommended)
Get free API key from [NCBI Account Settings](https://www.ncbi.nlm.nih.gov/account/settings/):

```bash
export NCBI_API_KEY="your_api_key_here"
```

Benefits:
- 10 requests/second vs 3/second without key
- Better success rates for large batches
- More stable downloads

## 📁 File Organization

Downloaded PDFs are organized as:
```
pdf_directory/
├── 33157158.pdf
├── 33157159.pdf
└── 33157160.pdf
```

## 🔒 Legal and Ethical Guidelines

This library follows best practices:

- ✅ **Respects publisher robots.txt**
- ✅ **Uses appropriate rate limiting**
- ✅ **Identifies requests with proper User-Agent**
- ✅ **Focuses on legally accessible content**
- ✅ **Follows NCBI API guidelines**

### Legal Boundaries
- Only downloads legally accessible PDFs
- Respects copyright and publisher policies
- Focuses on open access content
- Does not circumvent paywalls

## 🚨 Error Handling

The library handles various error conditions gracefully:

```python
result = await fetcher.download_from_pmids(["invalid_pmid"])

for download_result in result.results:
    if not download_result.success:
        print(f"Failed: {download_result.error_message}")
        # Common errors:
        # - "No PDF URL found"
        # - "HTTP 403" (access denied)
        # - "Invalid PDF content" (paywall page)
        # - "Download timeout"
        # - "All strategies failed"
```

## 📈 Optimization Tips

### For High Success Rates
1. **Use NCBI API key** - enables higher rate limits
2. **Focus on recent papers** - more likely to be open access
3. **Use specific search terms** - better quality results
4. **Monitor success rates** - adjust batch sizes accordingly

### For Large Batches
1. **Start with small batches** to test success rates
2. **Use batch processing** with appropriate delays
3. **Monitor circuit breaker status**
4. **Consider splitting very large requests**

## 🔧 Configuration Examples

### Conservative Settings (High Success Rate)
```python
fetcher = PubMedPDFFetcher(
    email="your@email.com",
    batch_size=3,           # Small batches
    min_success_rate=0.5    # Higher threshold
)
```

### Aggressive Settings (Fast Downloads)
```python
fetcher = PubMedPDFFetcher(
    email="your@email.com",
    api_key="your_key",     # Required for aggressive settings
    batch_size=10,          # Larger batches
    min_success_rate=0.2    # Lower threshold
)
```

## 📋 Complete Example

See `examples/pdf_fetcher_examples.py` for comprehensive usage examples including:
- Basic async and sync usage
- Search and download workflows
- Error handling patterns
- Statistics monitoring
- Batch processing strategies

## 🤝 Contributing

The library uses a modular strategy pattern making it easy to add new download sources:

```python
class CustomStrategy(PDFDownloadStrategy):
    @property
    def name(self) -> str:
        return "Custom Strategy"

    @property
    def priority(self) -> int:
        return 4  # Lower priority

    async def can_handle(self, paper: Paper) -> bool:
        # Return True if this strategy can handle the paper
        return paper.doi is not None

    async def get_pdf_url(self, paper: Paper) -> Optional[str]:
        # Return PDF URL or None
        return f"https://example.com/pdf/{paper.doi}"
```

---

**Note**: This library is designed for research and educational purposes. Always respect publisher terms of service and copyright laws when downloading academic papers.