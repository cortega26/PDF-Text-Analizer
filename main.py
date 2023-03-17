import requests
import PyPDF2
import re
import nltk
from collections import Counter
from nltk.corpus import stopwords
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from languages import languages


def download_pdf(url):
    """Download a PDF file from a given URL."""
    response = requests.get(url)
    response.raise_for_status()
    with open("temp.pdf", "wb") as f:
        f.write(response.content)


def pdf_to_text(pdf_file):
    """Convert a PDF file to text."""
    pdf = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in range(len(pdf.pages)):
        text += pdf.pages[page].extract_text()
    return text


def remove_stop_words(text, language):
    """Remove stop words from a given text for a given language."""
    try:
        stop_words = set(stopwords.words(language.lower()))
        words = nltk.word_tokenize(text)
        filtered_text = [word for word in words if word.casefold() not in stop_words]
        return ' '.join(filtered_text)
    except OSError:
        print(f"Stopwords not available for {language}.")
        return text


def count_words(text):
    """Count the occurrences of each word in a given text."""
    words = re.findall(r'\b\w+\b', text)
    word_counts = dict(Counter(words).most_common(10))
    return word_counts


def count_word_or_phrase(text, word_or_phrase):
    """Count the number of occurrences of a given word or phrase in a given text."""
    count = text.count(word_or_phrase)
    return count


def detect_language(text):
    """Detect the language of a given text."""
    try:
        language = detect(text)
        return language
    except LangDetectException:
        return None


def main(url, word_or_phrase):
    """
    Run the main program, which downloads a PDF file from a given URL, converts it to text,
    removes stop words, and counts the occurrences of the top 10 most common words and a given
    word or phrase in the text.
    """
    download_pdf(url)
    text = pdf_to_text("temp.pdf")
    language = detect_language(text)

    if language:
        filtered_text = remove_stop_words(text, language)
        word_counts = count_words(filtered_text)
        word_or_phrase_count = count_word_or_phrase(text, word_or_phrase)

        print(f"Language: {language}")
        print("Top 10 Words:")
        for word, count in word_counts.items():
            print(f"{word}: {count}")
        print(f"Occurrences of '{word_or_phrase}': {word_or_phrase_count}")
    else:
        print("Error: Language not detected.")


if __name__ == '__main__':
    url = "https://underpost.net/ir/pdf/interes/La_Rebelion_de_Atlas.pdf"
    word_or_phrase = "¿Quién es John Galt?"
    main(url, word_or_phrase)
