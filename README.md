# Web Scraping Search Engine Aggregator

This project aggregates search results from multiple search engines (Google and Bing) using web scraping and implements a simple but effective relevance scoring system.

## Features

- Multi-engine search aggregation using web scraping
- Custom relevance scoring algorithm
- Modern, responsive web interface
- Real-time search results
- Relevance scoring for each result

## Setup

1. Clone the repository:
```bash
git clone https://github.com/vishnus1793/llm_based_search_engine.git
cd llm_based_search_engine
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Enter your search query in the search box
2. Click the Search button or press Enter
3. View the aggregated and ranked results from multiple search engines
4. Click on any result to visit the source page

## How it Works

The search aggregator:
1. Scrapes search results from Google and Bing
2. Calculates relevance scores based on:
   - Title match (50% weight)
   - Snippet match (30% weight)
   - URL match (20% weight)
3. Ranks and displays the most relevant results

## Requirements

- Python 3.7+
- Internet connection for web scraping

## Note

This implementation uses web scraping to gather search results. Please be mindful of the search engines' terms of service and implement appropriate rate limiting in production use.
