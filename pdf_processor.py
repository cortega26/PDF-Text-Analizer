import functools
import io
import logging
import re
import string
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from languages import languages
from urllib.parse import urlparse

import nltk
import requests
from langdetect import detect, lang_detect_exception
import fitz

logging.basicConfig(level=logging.INFO)

class PdfProcessor:
    """Class to process PDF files."""
    def __init__(self, url: str) -> None:
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            raise ValueError("Invalid URL provided.")
        self.url = url
        self.pdf_text = ""
        self.language_code = None
        self.language_name = None
        self.pdf_content = None
        self.stop_words = set()

    def download_pdf(self, retries=3, backoff=2) -> None:
        """Download a PDF file from a given URL with retry mechanism."""
        with requests.Session() as session:
            for i in range(retries):
                try:
                    response = session.get(self.url, timeout=10)
                    response.raise_for_status()
                    self.pdf_content = io.BytesIO(response.content)
                    logging.info("PDF downloaded successfully.")
                    break
                except requests.exceptions.RequestException as e:
                    logging.error(f"Failed to download PDF: {e}")
                    logging.info("Retrying...")
                    time.sleep(backoff ** i)
            else:
                raise requests.exceptions.RequestException("Failed to download PDF after retries.")

    def convert_to_text(self) -> None:
        """Convert the downloaded PDF file to text using PyMuPDF."""
        if not self.pdf_content:
            raise ValueError("No PDF content to convert.")
        try:
            pdf_document = fitz.open(stream=self.pdf_content, filetype="pdf")
            with ThreadPoolExecutor() as executor:
                pages = [pdf_document.load_page(i) for i in range(len(pdf_document))]
                texts = executor.map(self.extract_text_from_page, pages)
                self.pdf_text = ''.join(texts)
            logging.info("PDF converted to text successfully.")
        except Exception as e:
            logging.error(f"Failed to convert PDF to text: {e}")
            raise
        finally:
            pdf_document.close()

    @staticmethod
    def extract_text_from_page(page) -> str:
        """Extract text from a single PDF page."""
        return page.get_text()

    def detect_language(self) -> None:
        """Detect the language of the PDF text."""
        if not self.pdf_text:
            raise ValueError("No text to detect language.")
        try:
            self.language_code = detect(self.pdf_text)
            self.language_name = languages.get(self.language_code.lower(), self.language_code)
            logging.info(f"Detected language: {self.language_name} ({self.language_code})")
        except lang_detect_exception.LangDetectException:
            self.language_code = None
            self.language_name = None
            logging.warning("Language detection failed.")

    def download_nltk_data(self) -> None:
        """Download required NLTK data."""
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
        except Exception as e:
            logging.error(f"Failed to download NLTK data: {e}")
            raise

    @staticmethod
    @functools.lru_cache(maxsize=10)
    def get_cached_stop_words(language_name: str) -> set:
        """Get stop words for the detected language and cache the results."""
        return set(nltk.corpus.stopwords.words(language_name))

    def get_stop_words(self) -> None:
        """Get stop words for the detected language."""
        if self.language_name:
            try:
                self.stop_words = PdfProcessor.get_cached_stop_words(self.language_name)
            except nltk.corpus.reader.exceptions.CorpusReadError:
                logging.warning(f"No stopwords available for {self.language_name}. Using empty set.")
                self.stop_words = set()
        else:
            self.stop_words = set()

    def remove_stop_words(self, text: str) -> str:
        """Remove stop words from a given text."""
        words = nltk.word_tokenize(text)
        filtered_text = ' '.join(word for word in words if word.casefold() not in self.stop_words)
        return filtered_text

    def count_words(self) -> dict:
        """Count the occurrences of each word in the PDF text, excluding punctuation."""
        clean_text = re.sub(rf'[{string.punctuation}]', '', self.pdf_text)
        words = nltk.word_tokenize(clean_text)
        words = [word.lower() for word in words]
        word_counts = Counter(word for word in words if word not in self.stop_words)
        return dict(word_counts.most_common(10))

    def count_word_or_phrase(self, word_or_phrase: str) -> int:
        """Count the number of occurrences of a given word or phrase in the PDF text."""
        try:
            return self.pdf_text.lower().count(word_or_phrase.lower())
        except Exception as e:
            logging.error(f"Failed to count word or phrase: {e}")
            return 0

    def extract_metadata(self) -> dict:
        """Extract metadata from the PDF."""
        if not self.pdf_content:
            raise ValueError("No PDF content to extract metadata.")
        try:
            pdf_document = fitz.open(stream=self.pdf_content, filetype="pdf")
            metadata = pdf_document.metadata
            logging.info("PDF metadata extracted successfully.")
            return metadata
        except Exception as e:
            logging.error(f"Failed to extract metadata: {e}")
            return {}
        finally:
            pdf_document.close()

    def format_results(self, results: dict) -> str:
        """Format the results for better readability."""
        formatted_results = [
            "\nPDF Analysis Results",
            "=" * 20,
            "\nMetadata:\n" + "\n".join([f"  {key}: {value}" for key, value in results['Metadata'].items()]),
            f"\nLanguage: {results['Language']} ({results['Language Code']})",
            "\nTop 10 Words:",
            "\n".join([f"  {word}: {count}" for word, count in results['Top 10 Words'].items()]),
            f"\nOccurrences of '{results['Search Term']}': {results['Occurrences']}",
        ]
        return "\n".join(formatted_results)

    def main(self, word_or_phrase: str) -> str:
        """
        Run the main program, which downloads a PDF file from a given URL, converts it to text, removes stop
        words, and counts the occurrences of the top 10 most common words and a given word or phrase in the text.
        Returns: str: A formatted string containing the results.
        """
        try:
            self.download_pdf()
            self.convert_to_text()
            self.detect_language()

            if self.language_code is None:
                raise Exception("Language not detected.")

            self.download_nltk_data()
            self.get_stop_words()

            word_counts = self.count_words()
            word_or_phrase_count = self.count_word_or_phrase(word_or_phrase)
            metadata = self.extract_metadata()

            results = {
                "Metadata": metadata,
                "Language": self.language_name,
                "Language Code": self.language_code,
                "Top 10 Words": word_counts,
                "Search Term": word_or_phrase,
                "Occurrences": word_or_phrase_count
            }
            return self.format_results(results)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return f"Error: {str(e)}"

