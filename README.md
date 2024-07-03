# PDF Text Analyzer

PDF Text Analyzer is a Python class that downloads a PDF from a URL, converts it to text, and performs text analysis. The analysis includes detecting the language of the text, removing stopwords, counting word frequencies, searching for specific phrases, and extracting metadata.

## Requirements

To use this class, you need Python 3.x and the following libraries:

- `requests`
- `PyMuPDF` (fitz)
- `nltk`
- `langdetect`
  
You can install the required libraries using the following command:

```sh
pip install requests PyMuPDF nltk langdetect
```

## Usage

1. Import the `PdfProcessor` class from the script.
2. Create an instance of `PdfProcessor` with a PDF URL.
3. Call the `main` method with an optional search phrase.

Example:

```python
from pdf_processor import PdfProcessor

pdf_processor = PdfProcessor("https://example.com/document.pdf")
results = pdf_processor.main("example phrase")
print(results)
```

## Features

- **PDF Download**: Downloads PDFs with retry mechanism.
- **Text Extraction**: Extracts text using multiprocessing for efficiency.
- **Language Detection**: Detects the document language.
- **Stop Words Removal**: Removes stop words (with caching for optimization).
- **Word Frequency Counting**: Counts word frequencies.
- **Phrase Search**: Searches for specific phrases.
- **Metadata Extraction**: Extracts PDF metadata.
- **Multilingual Support**: Supports multiple languages (based on NLTK's stopwords corpus).

## Error Handling

The script includes comprehensive error handling for:

- PDF download failures
- Text extraction issues
- Language detection problems

## Installation

To install the PDF Text Analyzer, clone this repository and install the dependencies:

```sh
git clone https://github.com/cortega26/PDF-Text-Analizer.git
cd PDF-Text-Analyzer
pip install -r requirements.txt
```

## Contributing

If you want to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/your-feature).
3. Commit your changes (git commit -am 'Add new feature').
4. Push to the branch (git push origin feature/your-feature).
5. Submit a pull request.

For any bugs or feature requests, please open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
