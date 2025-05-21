# PDF to University MCQ Generator

> Created by [Ishaan Rai](https://github.com/Ishaan-Rai09)

A Streamlit app that generates university-level multiple-choice questions (MCQs) from PDF documents using the Groq API.

## Features

- Upload PDF files and extract text content
- Generate university-level MCQs based on the content
- Support for multiple PDF files
- Enhanced PDF extraction with OCR capabilities
- Customizable number of questions and difficulty levels
- Choose between different Groq AI models

## Local Installation

1. Clone this repository
2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
3. For OCR capabilities, install system dependencies:
   - **Windows**:
     - Install [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)
     - Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
     - Add both to your PATH
   - **Linux**:
     ```
     sudo apt-get update
     sudo apt-get install -y poppler-utils tesseract-ocr libtesseract-dev
     ```
   - **macOS**:
     ```
     brew install poppler tesseract
     ```

4. Run the app:
   ```
   streamlit run app.py
   ```

## Streamlit Cloud Deployment

To deploy this app on Streamlit Cloud with full OCR capabilities:

1. Make sure your repository includes:
   - `app.py`
   - `requirements.txt` (with all Python dependencies)
   - `packages.txt` (with all system dependencies)
   - `setup.sh` (optional verification script)

2. The `packages.txt` file should contain:
   ```
   poppler-utils
   tesseract-ocr
   libtesseract-dev
   libleptonica-dev
   pkg-config
   libpng-dev
   libjpeg-dev
   libtiff-dev
   zlib1g-dev
   ```

3. Deploy to Streamlit Cloud by connecting your GitHub repository.

## Usage

1. Enter your Groq API key and validate it
2. Upload one or more PDF files
3. Set the number of questions and difficulty level
4. Click "Generate MCQs"
5. View and download the generated questions

## API Key

This app requires a Groq API key to function. Get your key from [Groq Console](https://console.groq.com/keys).

## License

[MIT License](LICENSE)

## Attribution

This project was created by [Your Full Name](https://github.com/your-username).  
Original repository: [github.com/your-username/your-repo](https://github.com/your-username/your-repo)

If you use, modify, or distribute this code, please maintain proper attribution to the original author and repository.

---

© 2024 [Your Full Name]. All Rights Reserved. 