"""
Enhanced PDF Processing System
A comprehensive solution for PDF analysis, text extraction, and content processing
with advanced features including caching, async operations, and content analysis.
Version: 2.1.1
"""

import asyncio
import dataclasses
import hashlib
import io
import json
import logging
import multiprocessing
import os
import re
import string
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set, List, Any, TypeVar, Generic, Union

import aiohttp
import nltk
import numpy as np
from langdetect import detect, LangDetectException
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer

# Type variables
T = TypeVar('T')
CacheKey = TypeVar('CacheKey')
CacheValue = TypeVar('CacheValue')

# Custom logging filter for correlation ID
class CorrelationFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = '-'
        return True

# Configure logging with correlation ID
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] %(message)s'
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.addFilter(CorrelationFilter())
logger.handlers = [handler]

@dataclass
class ProcessingStatistics:
    """Statistics about the PDF processing operation."""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_pages: int = 0
    processed_pages: int = 0
    total_words: int = 0
    processing_time: float = 0.0
    memory_used: float = 0.0

@dataclass
class PdfMetadata:
    """Enhanced metadata structure for PDF documents."""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    file_size: int = 0
    page_count: int = 0
    encrypted: bool = False
    permissions: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format."""
        return dataclasses.asdict(self)

class ProcessingError(Exception):
    """Custom exception for PDF processing errors."""
    pass

class Cache(Generic[CacheKey, CacheValue], ABC):
    """Protocol defining the cache interface."""
    
    @abstractmethod
    def get(self, key: CacheKey) -> Optional[CacheValue]:
        """Retrieve a value from cache."""
        pass
    
    @abstractmethod
    def put(self, key: CacheKey, value: CacheValue) -> None:
        """Store a value in cache."""
        pass
    
    @abstractmethod
    def invalidate(self, key: CacheKey) -> None:
        """Remove a value from cache."""
        pass

class SimpleMemoryCache(Cache[str, Any]):
    """Simple in-memory cache implementation with TTL."""
    
    def __init__(self, ttl_seconds: int = 3600):
        self._cache: Dict[str, tuple[Any, float]] = {}
        self._ttl_seconds = ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        value, timestamp = self._cache[key]
        if time.time() - timestamp > self._ttl_seconds:
            del self._cache[key]
            return None
        return value
    
    def put(self, key: str, value: Any) -> None:
        self._cache[key] = (value, time.time())
    
    def invalidate(self, key: str) -> None:
        self._cache.pop(key, None)

class ContentAnalyzer:
    """Analyzes text content using various NLP techniques."""
    
    def __init__(self, language: str):
        self.language = language
        self.vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=1000,
            ngram_range=(1, 2)
        )
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[tuple[str, float]]:
        """Extract important keywords using TF-IDF."""
        try:
            tfidf_matrix = self.vectorizer.fit_transform([text])
            feature_names = self.vectorizer.get_feature_names_out()
            scores = zip(feature_names, tfidf_matrix.toarray()[0])
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            return sorted_scores[:top_n]
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []
    
    def calculate_readability_score(self, text: str) -> float:
        """Calculate text readability using Flesch Reading Ease."""
        try:
            words = text.split()
            sentences = nltk.sent_tokenize(text)
            
            if not words or not sentences:
                return 0.0
            
            word_count = len(words)
            sentence_count = len(sentences)
            syllable_count = sum(self._count_syllables(word) for word in words)
            
            score = 206.835 - 1.015 * (word_count / sentence_count)
            if word_count > 0:
                score -= 84.6 * (syllable_count / word_count)
            
            return round(max(0.0, min(100.0, score)), 2)
        except Exception as e:
            logger.error(f"Readability calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def _count_syllables(word: str) -> int:
        """Count syllables in a word."""
        word = word.lower().strip()
        if not word:
            return 0
            
        count = 0
        vowels = set("aeiouy")
        prev_char = None
        
        for char in word:
            if char in vowels and (prev_char is None or prev_char not in vowels):
                count += 1
            prev_char = char
            
        if word.endswith(('e', 'es', 'ed')) and count > 1:
            count -= 1
        
        return max(1, count)

class PdfProcessor:
    """Enhanced PDF processor with advanced features."""
    
    # Configuration constants
    MAX_PDF_SIZE = 100 * 1024 * 1024  # 100MB
    DOWNLOAD_TIMEOUT = 30  # seconds
    MAX_RETRIES = 3
    BACKOFF_FACTOR = 2
    CACHE_SIZE = 100
    CHUNK_SIZE = 8192
    ALLOWED_CONTENT_TYPES = {'application/pdf', 'application/x-pdf'}
    
    def __init__(
        self,
        pdf_url: Optional[str] = None,
        cache: Optional[Cache] = None,
        max_workers: Optional[int] = None,
        storage_path: Optional[Path] = None
    ):
        """
        Initialize the PDF processor.
        
        Args:
            pdf_url: Optional URL of the PDF to process.
            cache: Optional cache instance.
            max_workers: Maximum number of threads for text extraction.
            storage_path: Path to store temporary data.
        """
        self.url = pdf_url
        self.cache = cache or SimpleMemoryCache()
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) * 4)
        self.storage_path = storage_path or Path.home() / ".pdfprocessor"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.stats = ProcessingStatistics()
        self._correlation_id = '-'
        
        # Initialize NLTK data at startup
        self._ensure_nltk_data()
    
    def _ensure_nltk_data(self) -> None:
        """Ensure all required NLTK data is downloaded."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading required NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
    
    def _get_nltk_stopwords(self, language: str) -> set:
        """
        Map detected language code to NLTK language name and return its stopwords.
        If the mapping is not available, it attempts to use the provided language directly.
        """
        lang_mapping = {
            "en": "english",
            "fr": "french",
            "de": "german",
            "es": "spanish",
            "it": "italian",
            "pt": "portuguese",
            "nl": "dutch",
            "sv": "swedish",
            "no": "norwegian",
            "fi": "finnish",
            "ru": "russian"
            # Add more mappings as needed
        }
        nltk_lang = lang_mapping.get(language, language)
        try:
            return set(nltk.corpus.stopwords.words(nltk_lang))
        except LookupError:
            logger.warning(f"Stopwords not available for {nltk_lang}, using empty set")
            return set()
    
    async def process_url(self, url: str, word_or_phrase: str) -> Dict[str, Any]:
        """Process a PDF from URL."""
        self._correlation_id = hashlib.md5(url.encode()).hexdigest()[:8]
        self.stats.start_time = time.time()
        
        try:
            # Check cache
            cache_key = f"pdf_analysis_{hashlib.md5(url.encode()).hexdigest()}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info("Returning cached result")
                return cached_result
            
            # Download and process
            content = await self._download_pdf(url)
            text, metadata = await self._process_pdf(content)
            
            # Analyze content
            analysis_results = await self._analyze_content(text, word_or_phrase)
            
            # Update statistics
            self.stats.end_time = time.time()
            self.stats.processing_time = self.stats.end_time - self.stats.start_time
            self.stats.total_words = analysis_results.get('word_count', 0)
            
            # Format timestamps for presentation
            stats_dict = dataclasses.asdict(self.stats)
            stats_dict['start_time'] = datetime.fromtimestamp(self.stats.start_time).isoformat()
            stats_dict['end_time'] = datetime.fromtimestamp(self.stats.end_time).isoformat() if self.stats.end_time else None
            
            # Prepare results without the full text
            results = {
                "metadata": metadata.to_dict(),
                "analysis": analysis_results,
                "statistics": stats_dict
            }
            
            # Cache results
            self.cache.put(cache_key, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise ProcessingError(f"Failed to process PDF: {str(e)}")
    
    async def _download_pdf(self, url: str) -> bytes:
        """Download PDF with retry logic and validate content type."""
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.MAX_RETRIES):
                try:
                    async with session.get(url, timeout=self.DOWNLOAD_TIMEOUT) as response:
                        response.raise_for_status()
                        # Validate content type
                        content_type = response.headers.get("Content-Type", "").split(";")[0]
                        if content_type not in self.ALLOWED_CONTENT_TYPES:
                            raise ProcessingError(f"Invalid content type: {content_type}")
                        
                        content = await response.read()
                        
                        if len(content) > self.MAX_PDF_SIZE:
                            raise ProcessingError("PDF file too large")
                        
                        return content
                        
                except Exception as e:
                    if attempt == self.MAX_RETRIES - 1:
                        raise ProcessingError(f"Failed to download PDF: {str(e)}")
                    await asyncio.sleep(self.BACKOFF_FACTOR ** attempt)
    
    async def _process_pdf(self, content: bytes) -> tuple[str, PdfMetadata]:
        """Process PDF content."""
        def process_in_thread() -> tuple[str, PdfMetadata]:
            with fitz.open(stream=content, filetype="pdf") as doc:
                # Extract metadata
                raw_metadata = doc.metadata
                metadata = PdfMetadata(
                    title=raw_metadata.get('title'),
                    author=raw_metadata.get('author'),
                    subject=raw_metadata.get('subject'),
                    keywords=raw_metadata.get('keywords'),
                    creator=raw_metadata.get('creator'),
                    producer=raw_metadata.get('producer'),
                    creation_date=raw_metadata.get('creationDate'),
                    modification_date=raw_metadata.get('modDate'),
                    file_size=len(content),
                    page_count=len(doc),
                    encrypted=doc.is_encrypted,
                    permissions={
                        'print': bool(doc.permissions & fitz.PDF_PERM_PRINT),
                        'modify': bool(doc.permissions & fitz.PDF_PERM_MODIFY),
                        'copy': bool(doc.permissions & fitz.PDF_PERM_COPY),
                        'annotate': bool(doc.permissions & fitz.PDF_PERM_ANNOTATE)
                    }
                )
                
                # Extract text concurrently using ThreadPoolExecutor
                self.stats.total_pages = len(doc)
                texts = []
                
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [
                        executor.submit(doc[page_num].get_text)
                        for page_num in range(len(doc))
                    ]
                    
                    for future in futures:
                        try:
                            text = future.result()
                            texts.append(text)
                            self.stats.processed_pages += 1
                        except Exception as e:
                            logger.error(f"Error extracting text from page: {e}")
                
                return ''.join(texts), metadata
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, process_in_thread)
    
    async def _analyze_content(self, text: str, word_or_phrase: str) -> Dict[str, Any]:
        """Perform content analysis with improved search term counting and output formatting."""
        try:
            # Detect language using a snippet of text for efficiency
            language = detect(text[:10000]) if text.strip() else "unknown"
            
            # Initialize analyzer
            analyzer = ContentAnalyzer(language)
            
            loop = asyncio.get_event_loop()
            
            def analyze_in_thread():
                try:
                    words = nltk.word_tokenize(text.lower())
                    
                    # Use the helper to get the proper stopwords
                    stop_words = self._get_nltk_stopwords(language)
                    
                    # Count exact occurrences of the search term
                    search_term_count = len(re.findall(
                        rf'\b{re.escape(word_or_phrase.lower())}\b', 
                        text.lower()
                    ))
                    
                    # Extract keywords and find matching ones
                    keywords = analyzer.extract_keywords(text)
                    matching_keywords = [
                        (kw, score) for kw, score in keywords
                        if word_or_phrase.lower() in kw.lower()
                    ]
                    
                    # Filter out non-alphabetic tokens and stopwords for top words
                    top_words = dict(Counter(
                        word for word in words
                        if word.isalpha() and word not in stop_words
                    ).most_common(10))
                    
                    # Create a preview of the text (first 500 characters)
                    text_preview = text[:500] + "..." if len(text) > 500 else text
                    
                    return {
                        'language': language,
                        'word_count': len(words),
                        'character_count': len(text),
                        'sentence_count': len(nltk.sent_tokenize(text)),
                        'search_term_count': search_term_count,
                        'keywords': keywords,
                        'matching_keywords': matching_keywords,
                        'readability_score': analyzer.calculate_readability_score(text),
                        'text_preview': text_preview,
                        'top_words': top_words
                    }
                except Exception as e:
                    logger.error(f"Error in content analysis thread: {e}")
                    raise
            
            return await loop.run_in_executor(None, analyze_in_thread)
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return {
                'language': 'unknown',
                'word_count': len(text.split()),
                'character_count': len(text),
                'sentence_count': 0,
                'search_term_count': text.lower().count(word_or_phrase.lower()),
                'keywords': [],
                'matching_keywords': [],
                'readability_score': 0.0,
                'text_preview': text[:500] + "..." if len(text) > 500 else text,
                'top_words': {}
            }
    
    def main(self, word_or_phrase: str) -> Dict[str, Any]:
        """
        Synchronous wrapper to process the PDF using the stored URL.
        
        Args:
            word_or_phrase: The phrase to search in the PDF.
            
        Returns:
            A dictionary with metadata, analysis, and processing statistics.
        """
        if not self.url:
            raise ValueError("PDF URL not provided.")
        return asyncio.run(self.process_url(self.url, word_or_phrase))
    
    def __call__(self, word_or_phrase: str) -> Dict[str, Any]:
        """
        Allow the instance to be called directly as a function.
        
        Args:
            word_or_phrase: The phrase to search in the PDF.
            
        Returns:
            A dictionary with metadata, analysis, and processing statistics.
        """
        return self.main(word_or_phrase)

class PdfBatch:
    """Handle batch processing of multiple PDFs."""
    
    def __init__(self, processor: PdfProcessor):
        self.processor = processor
        self.results: Dict[str, Any] = {}
        self.errors: Dict[str, str] = {}
    
    async def process_urls(self, urls: List[str], word_or_phrase: str) -> Dict[str, Any]:
        """Process multiple URLs concurrently."""
        tasks = []
        for url in urls:
            task = asyncio.create_task(self._process_single_url(url, word_or_phrase))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        return {
            'results': self.results,
            'errors': self.errors,
            'summary': self._generate_summary()
        }
    
    async def _process_single_url(self, url: str, word_or_phrase: str) -> None:
        """Process a single URL."""
        try:
            result = await self.processor.process_url(url, word_or_phrase)
            self.results[url] = result
        except Exception as e:
            self.errors[url] = str(e)
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate processing summary."""
        total_docs = len(self.results) + len(self.errors)
        return {
            'total_processed': len(self.results),
            'total_errors': len(self.errors),
            'success_rate': (len(self.results) / total_docs * 100) if total_docs > 0 else 0,
            'average_processing_time': np.mean([
                result['statistics']['processing_time']
                for result in self.results.values()
            ]) if self.results else 0,
            'total_pages_processed': sum(
                result['metadata']['page_count']
                for result in self.results.values()
            )
        }

class PdfSearchEngine:
    """Search engine for processed PDF content."""
    
    def __init__(self):
        self.index = defaultdict(list)
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
    
    def add_document(self, url: str, analysis_results: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """Add a document to the search index using analysis results."""
        doc_id = hashlib.md5(url.encode()).hexdigest()
        
        # Extract text content from analysis results (using the preview)
        content = analysis_results.get('text_preview', '')
        
        self.documents[doc_id] = {
            'url': url,
            'metadata': metadata,
            'content': content,
            'keywords': analysis_results.get('keywords', []),
            'matching_keywords': analysis_results.get('matching_keywords', []),
            'search_term_count': analysis_results.get('search_term_count', 0),
            'language': analysis_results.get('language', 'unknown')
        }
        
        # Index words from content
        words = set(word.lower() for word in nltk.word_tokenize(content))
        for word in words:
            self.index[word].append(doc_id)
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for documents matching query."""
        query_words = set(word.lower() for word in nltk.word_tokenize(query))
        
        # Calculate document scores
        doc_scores = defaultdict(float)
        for word in query_words:
            matching_docs = self.index.get(word, [])
            word_score = 1.0 / (len(matching_docs) if matching_docs else 1.0)
            for doc_id in matching_docs:
                doc_scores[doc_id] += word_score
        
        # Sort documents by score
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        # Format results
        results = []
        for doc_id, score in sorted_docs:
            doc = self.documents[doc_id]
            snippet = self._generate_snippet(doc['content'], query_words)
            
            results.append({
                'url': doc['url'],
                'metadata': doc['metadata'],
                'relevance_score': round(score, 3),
                'snippet': snippet,
                'language': doc['language'],
                'search_term_count': doc['search_term_count'],
                'matching_keywords': [
                    {'keyword': kw, 'score': score}
                    for kw, score in doc['matching_keywords']
                ]
            })
        
        return results
    
    def _generate_snippet(self, content: str, query_words: Set[str], 
                         context_words: int = 10) -> str:
        """Generate a relevant text snippet containing query words."""
        words = content.split()
        best_snippet = ""
        max_matches = 0
        
        # Slide a window over the text to find the best matching context
        for i in range(len(words)):
            window = words[i:i + context_words * 2]
            if not window:
                break
            
            # Count query word matches in this window
            matches = sum(1 for word in window 
                         if word.lower() in query_words)
            
            # Update best snippet if this window has more matches
            if matches > max_matches:
                max_matches = matches
                best_snippet = ' '.join(window)
        
        # Add ellipsis if we have a snippet
        return f"{best_snippet}..." if best_snippet else ""

def print_pdf_summary(results: Dict[str, Any]) -> None:
    """Print a formatted summary of single PDF processing results."""
    metadata = results.get("metadata", {})
    analysis = results.get("analysis", {})
    statistics = results.get("statistics", {})
    print("\n--- PDF Metadata ---")
    for key, value in metadata.items():
        print(f"{key.title()}: {value}")
    print("\n--- PDF Analysis ---")
    print(f"Language: {analysis.get('language', 'N/A')}")
    print(f"Word Count: {analysis.get('word_count', 'N/A')}")
    print(f"Character Count: {analysis.get('character_count', 'N/A')}")
    print(f"Sentence Count: {analysis.get('sentence_count', 'N/A')}")
    print(f"Search Term Count: {analysis.get('search_term_count', 'N/A')}")
    print(f"Readability Score: {analysis.get('readability_score', 'N/A')}")
    print("Keywords:")
    for kw, score in analysis.get("keywords", []):
        print(f"  {kw}: {score:.2f}")
    print("Top Words:")
    for word, count in analysis.get("top_words", {}).items():
        print(f"  {word}: {count}")
    print("\nText Preview:")
    print(analysis.get("text_preview", ""))
    print("\n--- Processing Statistics ---")
    for key, value in statistics.items():
        print(f"{key.replace('_',' ').title()}: {value}")

def print_batch_summary(batch_results: Dict[str, Any]) -> None:
    """Print a formatted summary for batch processing results."""
    summary = batch_results.get("summary", {})
    print("\n=== Batch Processing Summary ===")
    print(f"Total Processed: {summary.get('total_processed')}")
    print(f"Total Errors: {summary.get('total_errors')}")
    print(f"Success Rate: {summary.get('success_rate'):.2f}%")
    print(f"Average Processing Time: {summary.get('average_processing_time'):.2f} seconds")
    print(f"Total Pages Processed: {summary.get('total_pages_processed')}")

def print_search_results(search_results: List[Dict[str, Any]]) -> None:
    """Print formatted search results."""
    print("\n=== Search Results ===")
    for result in search_results:
        metadata = result.get("metadata", {})
        print("\n----------------------------------------")
        print(f"Title: {metadata.get('title', 'N/A')}")
        print(f"Author: {metadata.get('author', 'N/A')}")
        print(f"URL: {result.get('url', 'N/A')}")
        print(f"Relevance Score: {result.get('relevance_score', 'N/A')}")
        print(f"Snippet: {result.get('snippet', '')}")
        print("----------------------------------------\n")

def setup_nltk_data() -> None:
    """Download required NLTK data."""
    required_packages = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
    for package in required_packages:
        try:
            nltk.download(package, quiet=True)
        except Exception as e:
            logger.error(f"Failed to download NLTK package {package}: {e}")

async def main():
    """Example usage of the PDF processor with improved output presentation."""
    # Setup
    setup_nltk_data()
    
    # Initialize processor with cache
    processor = PdfProcessor(
        pdf_url="https://antilogicalism.com/wp-content/uploads/2017/07/atlas-shrugged.pdf",
        cache=SimpleMemoryCache(ttl_seconds=3600),
        storage_path=Path.home() / '.pdfprocessor'
    )
    
    search_term = "Who is John Galt?"
    
    try:
        # Process single PDF
        print("\nProcessing single PDF...")
        results = await processor.process_url(processor.url, search_term)
        print_pdf_summary(results)
        
        # Process directory of PDFs (if a directory exists)
        print("\nProcessing directory of PDFs...")
        directory = Path("./pdfs")  # Replace with actual directory if needed
        if directory.exists():
            batch_results = await PdfBatch(processor).process_urls(
                [f'file://{pdf_file.absolute()}' for pdf_file in directory.glob('**/*.pdf')],
                search_term
            )
            print_batch_summary(batch_results)
        
        # Search example
        print("\nPerforming search...")
        search_engine = PdfSearchEngine()
        search_engine.add_document(processor.url, results['analysis'], results['metadata'])
        search_results = search_engine.search(search_term)
        print_search_results(search_results)
        
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Processing failed: {e}")

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
