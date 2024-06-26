import io
import re
import logging
import string
from collections import Counter

import nltk
import requests
from langdetect import detect, lang_detect_exception
from pypdf import PdfReader
from languages import languages

# Set up logging
logging.basicConfig(level=logging.INFO)


class PdfProcessor:
    """Class to process PDF files."""
    
    def __init__(self, url: str) -> None:
        self.url = url
        self.pdf_text = ""
        self.language_code = None
        self.pdf_content = None
        self.stop_words = set()

    def download_pdf(self) -> None:
        """Download a PDF file from a given URL."""
        try:
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()
            self.pdf_content = io.BytesIO(response.content)
            logging.info("PDF downloaded successfully.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download PDF: {e}")
            raise e

    def convert_to_text(self) -> None:
        """Convert the downloaded PDF file to text."""
        try:
            pdf_reader = PdfReader(self.pdf_content)
            self.pdf_text = ''.join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            logging.info("PDF converted to text successfully.")
        except Exception as e:
            logging.error(f"Failed to convert PDF to text: {e}")
            raise e

    def detect_language(self) -> None:
        """Detect the language of the PDF text."""
        try:
            self.language_code = detect(self.pdf_text)
            logging.info(f"Detected language: {self.language_code}")
        except lang_detect_exception.LangDetectException:
            self.language_code = None
            logging.warning("Language detection failed.")

    def download_nltk_data(self) -> None:
        """Download required NLTK data."""
        try:
            nltk.download('stopwords')
            nltk.download('punkt')
        except Exception as e:
            logging.error(f"Failed to download NLTK data: {e}")
            raise e

    def get_stop_words(self) -> None:
        """Get stop words for the detected language."""
        if self.language_code:
            language_name = languages.get(self.language_code.lower(), self.language_code)
            self.stop_words = set(nltk.corpus.stopwords.words(language_name))
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
        try:
            pdf_reader = PdfReader(self.pdf_content)
            metadata = pdf_reader.metadata
            logging.info("PDF metadata extracted successfully.")
            return metadata
        except Exception as e:
            logging.error(f"Failed to extract metadata: {e}")
            return {}

    def main(self, word_or_phrase: str) -> dict:
        """
        Run the main program, which downloads a PDF file from a given URL, converts it to text,
        removes stop words, and counts the occurrences of the top 10 most common words and a given
        word or phrase in the text.
        Returns: dict: A dictionary containing the results.
        """
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
            "Language": languages.get(self.language_code, self.language_code),
            "Top 10 Words": word_counts,
            f"Occurrences of '{word_or_phrase}'": word_or_phrase_count
        }
        return results

# Example usage
pdf_processor = PdfProcessor("https://antilogicalism.com/wp-content/uploads/2017/07/atlas-shrugged.pdf")
print(pdf_processor.main('Who is John Galt?'))
