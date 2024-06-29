import io
import re
from collections import Counter
import nltk
import requests
import PyPDF2
from langdetect import detect, lang_detect_exception
from languages import languages
import string

class PdfProcessor:
    """Class to process PDF files."""
    def __init__(self, url: str):
        self.url = url
        self.pdf_text = ""
        self.language_code = None
        self.pdf_content = None

    def download_pdf(self) -> None:
        """Download a PDF file from a given URL."""
        try:
            response = requests.get(self.url)
            response.raise_for_status()
            self.pdf_content = io.BytesIO(response.content)
        except requests.exceptions.RequestException as e:
            raise e

    def convert_to_text(self) -> None:
        """Convert the downloaded PDF file to text."""
        pdf_reader = PyPDF2.PdfReader(self.pdf_content)
        for page in range(len(pdf_reader.pages)):
            self.pdf_text += pdf_reader.pages[page].extract_text()
        self.detect_language()

    def detect_language(self) -> None:
        """Detect the language of the PDF text."""
        try:
            self.language_code = detect(self.pdf_text)
        except lang_detect_exception.LangDetectException:
            self.language_code = None

    def download_nltk_data(self) -> None:
        """Download required NLTK data."""
        if not nltk.corpus.stopwords.fileids():
            nltk.download('stopwords')
        nltk.download('punkt')

    def remove_stop_words(self) -> str:
        """Remove stop words from the PDF text."""
        if self.language_code is not None:
            self.download_nltk_data()
            language_name = languages.get(self.language_code.lower(), self.language_code)
            stop_words = set(nltk.corpus.stopwords.words(language_name))
            words = nltk.word_tokenize(self.pdf_text)
            filtered_text = ' '.join([word for word in words if word.casefold() not in stop_words])
            return filtered_text
        else:
            raise Exception("Language not detected.")

    def count_words(self, text: str) -> dict:
        """Count the occurrences of each word in the provided text, excluding punctuation."""
        # Remove all punctuation characters, except for apostrophes
        clean_text = re.sub(r'[^a-zA-Z0-9\s\']', '', text)

        # Tokenize the text
        words = nltk.word_tokenize(clean_text)

        # Convert words to lowercase
        words = [word.lower() for word in words]

        # Count word occurrences, ignoring stop words
        if self.language_code is not None:
            self.download_nltk_data()
            language_name = languages.get(self.language_code.lower(), self.language_code)
            stop_words = set(nltk.corpus.stopwords.words(language_name))
            word_counts = Counter(word for word in words if word not in stop_words)
        else:
            word_counts = Counter(words)

        return dict(word_counts.most_common(10))


    def count_word_or_phrase(self, text: str, word_or_phrase: str) -> int:
        """Count the number of occurrences of a given word or phrase in the provided text."""
        return text.count(word_or_phrase)

    def main(self, word_or_phrase: str) -> dict:
        """
        Run the main program, which downloads a PDF file from a given URL, converts it to text, removes stop words, and counts the occurrences of the top 10 most common words and a given word or phrase in the text.
        Returns: dict: A dictionary containing the results.
        """
        self.download_pdf()
        self.convert_to_text()
        if self.language_code is not None:
            filtered_text = self.remove_stop_words()
        else:
            raise Exception("Language not detected.")
        
        # Use filtered text for counting words and phrases
        word_counts = self.count_words(filtered_text)
        word_or_phrase_count = self.count_word_or_phrase(filtered_text, word_or_phrase)
        
        results = {
            "Language": languages.get(self.language_code, self.language_code),
            "Top 10 Words": word_counts,
            f"Occurrences of '{word_or_phrase}'": word_or_phrase_count
        }
        return results

# Example usage
pdf_processor = PdfProcessor("https://antilogicalism.com/wp-content/uploads/2017/07/atlas-shrugged.pdf")
print(pdf_processor.main('Who is John Galt?'))
