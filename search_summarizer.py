import os
from dotenv import load_dotenv
from pathlib import Path
import json
from datetime import datetime
from typing import List, Optional
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from serpapi import GoogleSearch
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
from dataclasses import dataclass

# Load environment variables
load_dotenv()

# Configure LLaMAIndex
Settings.llm = Ollama(model="llama3", request_timeout=60.0)
Settings.node_parser = SentenceSplitter(chunk_size=1024)
Settings.chunk_size = 1024

@dataclass
class SearchResult:
    url: str
    title: str
    content: str

class SearchSummarizer:
    def __init__(self):
        self.cache_dir = Path("search_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.search_data_dir = Path("search_data")
        self.search_data_dir.mkdir(exist_ok=True)
        self.session = None
        self.search_engines = {
            "google": self.search_google,
            "duckduckgo": self.search_duckduckgo
        }

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        # Clean up search data files
        for file in self.search_data_dir.glob("*.txt"):
            file.unlink()

    def _get_cache_file(self, query: str) -> Path:
        safe_query = "".join(c if c.isalnum() else "_" for c in query)
        return self.cache_dir / f"{safe_query[:50]}.json"

    async def _load_from_cache(self, query: str) -> Optional[List[SearchResult]]:
        cache_file = self._get_cache_file(query)
        if cache_file.exists():
            with open(cache_file, "r") as f:
                data = json.load(f)
                if datetime.now().timestamp() - data["timestamp"] < 86400:  # 1 day cache
                    return [SearchResult(**item) for item in data["results"]]
        return None

    async def _save_to_cache(self, query: str, results: List[SearchResult]):
        cache_file = self._get_cache_file(query)
        data = {
            "timestamp": datetime.now().timestamp(),
            "results": [{
                "url": r.url,
                "title": r.title,
                "content": r.content
            } for r in results]
        }
        with open(cache_file, "w") as f:
            json.dump(data, f)

    def _save_search_data(self, query: str, contents: List[str]):
        """Save content to files for LLaMAIndex processing"""
        # Clear previous files
        for file in self.search_data_dir.glob("*.txt"):
            file.unlink()
        
        # Save new content
        for i, content in enumerate(contents):
            with open(self.search_data_dir / f"result_{i}.txt", "w") as f:
                f.write(f"URL: {contents[i]['url']}\n\n{content}")

    async def search_google(self, query: str, num_results: int = 3) -> List[SearchResult]:
        """Search using Google via SerpAPI"""
        if not os.getenv("SERPAPI_KEY"):
            return []
        
        params = {
            "q": query,
            "api_key": os.getenv("SERPAPI_KEY"),
            "num": num_results
        }
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            return [SearchResult(
                url=r.get("link"),
                title=r.get("title"),
                content=""
            ) for r in results.get("organic_results", [])]
        except Exception as e:
            print(f"Google search error: {e}")
            return []

    async def search_duckduckgo(self, query: str, num_results: int = 3) -> List[SearchResult]:
        """Search using DuckDuckGo"""
        try:
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query, max_results=num_results)]
            return [SearchResult(
                url=r.get("href"),
                title=r.get("title"),
                content=""
            ) for r in results]
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []

    async def fetch_page_content(self, url: str) -> str:
        """Fetch and extract text from a webpage"""
        if not self.session:
            return ""
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            async with self.session.get(url, headers=headers, timeout=10) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(["script", "style", "nav", "footer", "iframe", "noscript"]):
                    element.decompose()
                
                # Get text from important tags
                text_parts = []
                for tag in ['h1', 'h2', 'h3', 'p']:
                    text_parts.extend([e.get_text(strip=True) for e in soup.find_all(tag)])
                
                return ' '.join(text_parts)[:10000]  # Limit to 10k chars
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return ""

    async def summarize_with_llamaindex(self, query: str, contents: List[dict]) -> str:
        """Summarize content using LLaMAIndex"""
        if not contents:
            return "No content available for summarization."
        
        self._save_search_data(query, contents)
        
        try:
            # Load and index the documents
            documents = SimpleDirectoryReader(str(self.search_data_dir)).load_data()
            index = VectorStoreIndex.from_documents(documents)
            
            # Create query engine
            query_engine = index.as_query_engine(
                similarity_top_k=3,
                response_mode="tree_summarize"
            )
            
            # Generate summary
            response = query_engine.query(
                f"Provide a comprehensive summary about {query} based on these sources. "
                "Include key facts, figures, and different perspectives. "
                "Structure the response with clear sections if appropriate."
            )
            
            return str(response)
        except Exception as e:
            print(f"LLaMAIndex summarization error: {e}")
            return f"Summary unavailable due to error: {str(e)}"

    async def search_and_summarize(self, query: str, num_results: int = 3) -> str:
        """Main function to search and summarize results"""
        # Check cache first
        cached = await self._load_from_cache(query)
        if cached:
            print("Using cached results")
            contents = [{"url": r.url, "content": r.content} for r in cached]
            return await self.summarize_with_llamaindex(query, contents)
        
        print(f"\nSearching for: '{query}'...")
        
        # Run searches in parallel
        search_tasks = [engine(query, num_results) for engine in self.search_engines.values()]
        search_results = await asyncio.gather(*search_tasks)
        
        # Combine and deduplicate results
        all_results = []
        seen_urls = set()
        for results in search_results:
            for result in results:
                if result.url and result.url not in seen_urls:
                    seen_urls.add(result.url)
                    all_results.append(result)
        
        if not all_results:
            return "No results found from search engines."
        
        print(f"\nFound {len(all_results)} results. Fetching content...")
        
        # Fetch content in parallel
        fetch_tasks = [self.fetch_page_content(result.url) for result in all_results]
        contents = await asyncio.gather(*fetch_tasks)
        
        # Update results with content
        for result, content in zip(all_results, contents):
            result.content = content
        
        # Save to cache
        valid_results = [r for r in all_results if r.content]
        await self._save_to_cache(query, valid_results)
        
        print("\nGenerating summary using LLaMAIndex...")
        return await self.summarize_with_llamaindex(
            query, 
            [{"url": r.url, "content": r.content} for r in valid_results]
        )

async def main():
    try:
        print("LLaMAIndex-Powered Search Summarizer")
        print("Enter your query (or 'quit' to exit):")
        
        async with SearchSummarizer() as summarizer:
            while True:
                query = input("\n> ").strip()
                if query.lower() in ('quit', 'exit'):
                    break
                if not query:
                    continue
                    
                summary = await summarizer.search_and_summarize(query)
                print("\n=== Summary ===")
                print(summary)
                print("\n=== End of Summary ===")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())