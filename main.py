import requests
import PyPDF2
import re
import nltk
from collections import Counter
from nltk.corpus import stopwords
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from languages import languages


# Download PDF from URL
def download_pdf(url):
    response = requests.get(url)
    response.raise_for_status()
    with open("temp.pdf", "wb") as f:
        f.write(response.content)


# Convert PDF to text
def pdf_to_text(pdf_file):
    pdf = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in range(len(pdf.pages)):
        text += pdf.pages[page].extract_text()
    return text


# Remove stop words
def remove_stop_words(text, language):
    for lang in nltk.corpus.stopwords.fileids():
        try:
            nltk.corpus.stopwords.words(lang)
        except LookupError:
            nltk.download('stopwords', quiet=True)
    language = languages.get(language).lower()
    try:
        stop_words = set(stopwords.words(language))
        words = nltk.word_tokenize(text)
        filtered_text = [word for word in words if word.casefold() not in stop_words]
        return ' '.join(filtered_text)
    except OSError:
        print(f"Stopwords not available for {language}.")
        return text


# Count word occurrences
def count_words(text):
    words = re.findall(r'\b\w+\b', text)
    word_counts = dict(Counter(words).most_common(10))
    return word_counts


# Count word or phrase occurrences
def count_word_or_phrase(text, word_or_phrase):
    count = text.count(word_or_phrase)
    return count


# Detect language of text
def detect_language(text):
    try:
        language = detect(text)
        return language
    except LangDetectException:
        return None


# Main function
def main(url, word_or_phrase):
    # Download PDF
    download_pdf(url)

    # Convert PDF to text
    text = pdf_to_text("temp.pdf")

    # Detect language
    language = detect_language(text)

    # Remove stop words
    if language is not None:
        filtered_text = remove_stop_words(text, language)
        word_counts = count_words(filtered_text)
        word_or_phrase_count = count_word_or_phrase(text, word_or_phrase)

        # Display results
        print(f"Language: {language}")
        print("Top 10 Words:")
        for word, count in list(word_counts.items()):
            print(f"{word}: {count}")
        print(f"Occurrences of '{word_or_phrase}': {word_or_phrase_count}")
    else:
        print("Error: Language not detected.")

# Example usage
if __name__ == '__main__':
    url = "https://underpost.net/ir/pdf/interes/La_Rebelion_de_Atlas.pdf"
    word_or_phrase = "¿Quién es John Galt?"
    main(url, word_or_phrase)
