import os
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from serpapi import GoogleSearch
from openai import OpenAI
from dotenv import load_dotenv
import re
from urllib.parse import quote_plus

load_dotenv()

class SearchAggregator:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def search_google(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search using Google via web scraping"""
        try:
            encoded_query = quote_plus(query)
            url = f"https://www.google.com/search?q={encoded_query}"
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'lxml')
            
            results = []
            for g in soup.find_all('div', class_='g')[:num_results]:
                anchors = g.find_all('a')
                if not anchors:
                    continue
                    
                link = anchors[0]['href']
                if not link.startswith('http'):
                    continue
                    
                title = g.find('h3')
                if not title:
                    continue
                    
                snippet = g.find('div', class_='VwiC3b')
                if not snippet:
                    continue
                    
                results.append({
                    'title': title.text,
                    'link': link,
                    'snippet': snippet.text,
                    'source': 'Google'
                })
                
            return results
        except Exception as e:
            print(f"Google search error: {e}")
            return []

    def search_bing(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search using Bing via web scraping"""
        try:
            encoded_query = quote_plus(query)
            url = f"https://www.bing.com/search?q={encoded_query}"
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'lxml')
            
            results = []
            for li in soup.find_all('li', class_='b_algo')[:num_results]:
                title_elem = li.find('h2')
                if not title_elem:
                    continue
                    
                link = title_elem.find('a')
                if not link:
                    continue
                    
                snippet = li.find('div', class_='b_caption')
                if not snippet:
                    continue
                    
                results.append({
                    'title': title_elem.text,
                    'link': link['href'],
                    'snippet': snippet.text,
                    'source': 'Bing'
                })
                
            return results
        except Exception as e:
            print(f"Bing search error: {e}")
            return []

    def calculate_relevance_score(self, query: str, result: Dict) -> float:
        """Calculate relevance score based on text matching and position"""
        score = 0.0
        query_terms = set(query.lower().split())
        
        # Title relevance (weight: 0.5)
        title_terms = set(result['title'].lower().split())
        title_match = len(query_terms.intersection(title_terms)) / len(query_terms)
        score += title_match * 0.5
        
        # Snippet relevance (weight: 0.3)
        snippet_terms = set(result['snippet'].lower().split())
        snippet_match = len(query_terms.intersection(snippet_terms)) / len(query_terms)
        score += snippet_match * 0.3
        
        # URL relevance (weight: 0.2)
        url_terms = set(result['link'].lower().split('/'))
        url_match = len(query_terms.intersection(url_terms)) / len(query_terms)
        score += url_match * 0.2
        
        return min(score * 100, 100)  # Convert to percentage and cap at 100

    def aggregate_search(self, query: str, num_results: int = 5) -> List[Dict]:
        """Aggregate results from all search engines and rank them"""
        all_results = []
        
        # Gather results from different search engines
        all_results.extend(self.search_google(query, num_results))
        all_results.extend(self.search_bing(query, num_results))
        
        # Calculate relevance scores and sort
        for result in all_results:
            result['relevance_score'] = self.calculate_relevance_score(query, result)
        
        # Sort by relevance score and return top results
        ranked_results = sorted(all_results, key=lambda x: x['relevance_score'], reverse=True)
        return ranked_results[:num_results] 