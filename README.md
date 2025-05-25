# Multi-Search Engine Summarizer

A Python tool that searches multiple engines, fetches web content, and generates AI-powered summaries using local LLM (Ollama).

![Project Demo](demo.gif) *(Optional: Add a demo GIF later)*

## Features

- **Multi-engine search**: Google (via SerpAPI) and DuckDuckGo
- **Local AI processing**: Uses Ollama with LLaMA3 for summarization
- **Intelligent caching**: Stores results for 24 hours
- **Parallel processing**: Fast content fetching using async/await
- **Privacy-focused**: No data leaves your machine (except search queries)

## Requirements

- Python 3.10+
- Ollama (with at least 8GB RAM for LLaMA3)
- SerpAPI key (optional, for Google searches)

## Installation

1. **Install Ollama**:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull llama3
