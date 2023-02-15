# PDF Text Analysis

This is a Python script that downloads a PDF from a URL, converts it to text, and performs text analysis. The text analysis includes detecting the language of the text, removing stopwords and counting the frequency of words and phrases. The script supports multiple languages.

## Requirements

- Python 3.x
- [requests](https://pypi.org/project/requests/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [nltk](https://pypi.org/project/nltk/)
- [langdetect](https://pypi.org/project/langdetect/)

## Usage

1. Run the script: `python3 pdf_analysis.py`
2. Enter the URL of the PDF you want to analyze
3. Enter the language of the text in the PDF
4. Enter a word or phrase to search for in the text (optional)
5. The script will then download the PDF, extract the text, and perform analysis on the text
6. The analysis includes word frequency counts, most common words, and a search for a specific word or phrase

## Supported Languages

- Arabic
- Azerbaijani
- Basque
- Bengali
- Catalan
- Chinese
- Danish
- Dutch
- English
- Finnish
- French
- German
- Greek
- Hebrew
- Hinglish
- Hungarian
- Indonesian
- Italian
- Kazakh
- Nepali
- Norwegian
- Portuguese
- Romanian
- Russian
- Slovene
- Spanish
- Swedish
- Tajik
- Turkish

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
