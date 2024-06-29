import io
import re
from collections import Counter
import nltk
import requests
import pypdf
from langdetect import detect, lang_detect_exception
from languages import languages
import string

class PdfProcessor:
    """Class to process PDF files."""
    
    def __init__(self, url: str) -> None:
        self.url = url
        self.pdf_text = ""
        self.language_code = None
        self.pdf_content = None

    def download_pdf(self) -> None:
        """Download a PDF file from a given URL."""
        try:
            with requests.get(self.url) as response:
                response.raise_for_status()
                self.pdf_content = io.BytesIO(response.content)
        except requests.exceptions.RequestException as e:
            print(f"Failed to download PDF: {e}")
            raise e

    def convert_to_text(self) -> None:
        """Convert the downloaded PDF file to text."""
        try:
            pdf_reader = pypdf.PdfReader(self.pdf_content)
            for page in range(len(pdf_reader.pages)):
                self.pdf_text += pdf_reader.pages[page].extract_text()
        except Exception as e:
            print(f"Failed to convert PDF to text: {e}")
            raise e

    def detect_language(self) -> None:
        """Detect the language of the PDF text."""
        try:
            self.language_code = detect(self.pdf_text)
        except lang_detect_exception.LangDetectException:
            self.language_code = None

    def download_nltk_data(self) -> None:
        """Download required NLTK data."""
        try:
            if 'stopwords' not in nltk.data.find('corpora') or 'punkt' not in nltk.data.find('tokenizers'):
                nltk.download('stopwords')
                nltk.download('punkt')
        except Exception as e:
            print(f"Failed to download NLTK data: {e}")
            raise e

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

    def count_words(self) -> dict:
        """Count the occurrences of each word in the PDF text, excluding punctuation."""
        clean_text = re.sub(r'[' + string.punctuation + ']', '', self.pdf_text)
        words = nltk.word_tokenize(clean_text)
        words = [word.lower() for word in words]
        
        if self.language_code is not None:
            self.download_nltk_data()
            language_name = languages.get(self.language_code.lower(), self.language_code)
            stop_words = set(nltk.corpus.stopwords.words(language_name))
            word_counts = Counter(word for word in words if word not in stop_words)
        else:
            word_counts = Counter(words)

        return dict(word_counts.most_common(10))

    def count_word_or_phrase(self, word_or_phrase: str) -> int:
        """Count the number of occurrences of a given word or phrase in the PDF text."""
        try:
            return self.pdf_text.count(word_or_phrase)
        except Exception as e:
            print(f"Failed to count word or phrase: {e}")
            return 0

    def main(self, word_or_phrase: str) -> dict:
        """
        Run the main program, which downloads a PDF file from a given URL, converts it to text, removes stop words, and counts the occurrences of the top 10 most common words and a given word or phrase in the text.
        Returns: dict: A dictionary containing the results.
        """
        self.download_pdf()
        self.convert_to_text()
        self.detect_language()
        
        if self.language_code is None:
            raise Exception("Language not detected.")
        
        word_counts = self.count_words()
        word_or_phrase_count = self.count_word_or_phrase(word_or_phrase)
        
        results = {
            "Language": languages.get(self.language_code, self.language_code),
            "Top 10 Words": word_counts,
            f"Occurrences of '{word_or_phrase}'": word_or_phrase_count
        }
        return results

# Example usage
pdf_processor = PdfProcessor("https://antilogicalism.com/wp-content/uploads/2017/07/atlas-shrugged.pdf")
print(pdf_processor.main('Who is John Galt?'))
