# SearchLite 🔍

A lightning-fast, asynchronous real-time Google Search API wrapper with built-in optimization for batch queries.

[![PyPI version](https://img.shields.io/badge/pypi-v1.0.0-blue.svg)](https://pypi.org/project/searchlite/)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features ✨

- 🚀 Real-time Google search results
- 🔄 Asynchronous batch searching
- 🎯 Optimized for multiple queries
- 🧹 Automatic duplicate removal
- 🎨 Optional progress animation
- 🔧 Configurable worker pool

## Installation 📦

```bash
pip install searchlite
```

## Quick Start 🚀

### Basic Search

```python
from searchlite import RealTimeGoogleSearchProvider

# Initialize the search provider
searcher = RealTimeGoogleSearchProvider()

# Single query search
results = searcher.search("Python programming", max_urls=5)
print(results)
```

### Batch Search

```python
# Multiple queries at once
queries = [
    "machine learning basics",
    "data science projects",
    "python best practices"
]

# Batch search with async execution
results = searcher.search_batch(queries, max_urls=10)
print(results)
```

## Advanced Usage 🔧

### Custom Configuration

```python
searcher = RealTimeGoogleSearchProvider(
    search_provider="google",  # Search engine to use
    chromedriver_path="/custom/path/chromedriver",  # Custom ChromeDriver path
    max_workers=4,  # Number of concurrent workers
    animation=True  # Enable progress animation
)
```

### Async Implementation

```python
import asyncio

async def main():
    searcher = RealTimeGoogleSearchProvider()
    queries = ["AI news", "Python updates", "Tech trends"]
    
    # Using the internal async method
    results = await searcher._async_batch_search(queries, max_urls=5)
    return results

# Run async function
results = asyncio.run(main())
```

## Features Explained 📚

### URL Processing
- Automatic hash fragment removal
- Duplicate URL filtering
- Configurable result limit
- Maintains original URL order

### Batch Processing
- Concurrent execution
- Memory efficient
- Automatic error handling
- Result aggregation

## Requirements 🛠️

- Python 3.7+
- ChromeDriver
- Required Python packages:
  - `selenium`
  - `asyncio`
  - `typing`

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- Inspired by the need for efficient real-time search capabilities
- Built with ❤️ for the Python community
- Special thanks to all contributors

## Support 💬

For support, please open an issue in the GitHub repository or contact the maintainers.

---

Made with ❤️ 