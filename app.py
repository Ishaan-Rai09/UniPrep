"""
PDF to University MCQ Generator
Copyright (c) 2024 Ishaan Rai

All rights reserved. This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

If you use, modify, or distribute this code, please maintain this attribution header.
"""

import os
import streamlit as st

# Set page configuration first - MUST be the first Streamlit command
st.set_page_config(
    page_title="PDF to MCQ Generator",
    page_icon="📝",
    layout="wide"
)

import PyPDF2
import tempfile
from dotenv import load_dotenv
from groq import Groq
from typing import List, Dict, Tuple
from config import get_api_key, get_groq_api_key, get_openai_api_key, get_ollama_host
import pdfplumber
import io
import re
import time
import logging

# For OpenAI API integration
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# For Ollama integration
try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('pdf_mcq_generator')

# Load environment variables
load_dotenv()

# Initialize session state for config
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = ""
    st.session_state.groq_api_key_valid = False
    st.session_state.groq_client = None
    
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
    st.session_state.openai_api_key_valid = False
    st.session_state.openai_client = None
    
if 'ollama_host' not in st.session_state:
    st.session_state.ollama_host = "http://localhost:11434"
    st.session_state.ollama_host_valid = False

if 'active_provider' not in st.session_state:
    st.session_state.active_provider = "groq"  # Default provider

if 'api_key_valid' not in st.session_state:
    st.session_state.api_key_valid = False  # Legacy for backward compatibility
    
# For attribution
st.session_state.author_signature = "github.com/your-username/your-repo"
st.session_state.creation_date = "2024"

# Get Groq API key
GROQ_API_KEY = get_api_key()

# Selected model
MODEL = "llama3-70b-8192"
CHUNK_SIZE = 4000

# OCR availability flags
OCR_AVAILABLE = False
POPPLER_INSTALLED = False
TESSERACT_INSTALLED = False

# Try to import libraries for OCR functionality
ocr_messages = []
try:
    from pdf2image import convert_from_bytes
    try:
        # Test if poppler is installed
        test_pdf = io.BytesIO(b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 8 >>\nstream\nBT /F1 12 Tf (Test) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\n0000000198 00000 n\ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n256\n%%EOF\n")
        images = convert_from_bytes(test_pdf.read(), first_page=1, last_page=1)
        POPPLER_INSTALLED = True
        logger.info("Poppler is installed and working correctly!")
        ocr_messages.append("✅ Poppler is installed and working correctly! OCR extraction will be available.")
    except Exception as e:
        POPPLER_INSTALLED = False
        logger.error(f"Poppler not installed or not properly configured. Error: {str(e)}")
        ocr_messages.append(f"⚠️ Poppler not installed or not properly configured. Error: {str(e)}")
        # Check if this is likely a cloud deployment
        import platform
        if platform.system() == "Linux" and os.path.exists("/home/streamlit"):
            ocr_messages.append("🔄 Cloud deployment detected. If this is Streamlit Cloud, make sure you have a packages.txt file with 'poppler-utils' in it.")
        else:
            ocr_messages.append("👉 For Windows, download from [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/). Extract the ZIP and add the bin folder to your PATH.")
    
    import pytesseract
    try:
        # Test if tesseract is installed
        pytesseract.get_tesseract_version()
        TESSERACT_INSTALLED = True
        logger.info("Tesseract is installed and working correctly!")
        if POPPLER_INSTALLED:
            OCR_AVAILABLE = True
            logger.info("OCR is fully available with both Poppler and Tesseract!")
    except Exception as e:
        TESSERACT_INSTALLED = False
        logger.error(f"Tesseract not installed or not properly configured. Error: {str(e)}")
        ocr_messages.append(f"⚠️ Tesseract not installed or not properly configured. Error: {str(e)}")
        # Check if this is likely a cloud deployment
        import platform
        if platform.system() == "Linux" and os.path.exists("/home/streamlit"):
            ocr_messages.append("🔄 Cloud deployment detected. If this is Streamlit Cloud, make sure you have a packages.txt file with 'tesseract-ocr' in it.")
        else:
            ocr_messages.append("👉 Download from [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki). During installation, check 'Add to PATH' option.")
except ImportError as e:
    OCR_AVAILABLE = False
    logger.error(f"OCR libraries not installed. Error: {str(e)}")
    ocr_messages.append(f"⚠️ OCR libraries not installed. Error: {str(e)}")
    ocr_messages.append("👉 Run: pip install pytesseract pdf2image")

def validate_api_key(api_key):
    """Validate the Groq API key by making a simple test request."""
    if not api_key or len(api_key) < 10:  # Basic length check
        return False
        
    try:
        # Initialize a test client
        test_client = Groq(api_key=api_key)
        
        # Make a minimal API call to test the key
        response = test_client.chat.completions.create(
            model="llama3-8b-8192",  # Using smaller model for faster validation
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
        )
        
        # If we got a response without error, the key is valid
        return True
    except Exception as e:
        logger.error(f"Groq API key validation failed: {str(e)}")
        return False

def validate_openai_api_key(api_key):
    """Validate the OpenAI API key by making a simple test request."""
    if not api_key or len(api_key) < 10:  # Basic length check
        return False
        
    if not OPENAI_AVAILABLE:
        logger.warning("OpenAI validation skipped - OpenAI package not installed")
        return False
        
    try:
        # Initialize a test client
        client = OpenAI(api_key=api_key)
        
        # Make a minimal API call to test the key
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using smaller model for faster validation
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
        )
        
        # If we got a response without error, the key is valid
        return True
    except Exception as e:
        logger.error(f"OpenAI API key validation failed: {str(e)}")
        return False

def validate_ollama_host(host_url):
    """Validate the Ollama host by checking connectivity and available models."""
    if not host_url:  # Basic check
        return False
        
    if not OLLAMA_AVAILABLE:
        logger.warning("Ollama validation skipped - requests package not installed")
        return False
        
    try:
        # Make a request to the Ollama API to list models
        response = requests.get(f"{host_url}/api/tags", timeout=5)
        
        # Check if the request was successful
        if response.status_code == 200:
            return True
        else:
            logger.error(f"Ollama API returned status code: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Ollama host validation failed: {str(e)}")
        return False

def extract_text_with_pypdf2(pdf_file) -> str:
    """Extract text using PyPDF2."""
    start_time = time.time()
    logger.info(f"Starting PyPDF2 extraction for {pdf_file.name}")
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        elapsed = time.time() - start_time
        words = len(text.split())
        logger.info(f"PyPDF2 extraction completed: {words} words in {elapsed:.2f} seconds")
        return text
    except Exception as e:
        logger.warning(f"PyPDF2 extraction issue: {str(e)}")
        st.warning(f"PyPDF2 extraction issue: {str(e)}")
        return ""

def extract_text_with_pdfplumber(pdf_file) -> str:
    """Extract text using pdfplumber which handles more complex PDFs."""
    start_time = time.time()
    logger.info(f"Starting pdfplumber extraction for {pdf_file.name}")
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            elapsed = time.time() - start_time
            words = len(text.split())
            logger.info(f"PDFPlumber extraction completed: {words} words in {elapsed:.2f} seconds")
            return text
    except Exception as e:
        logger.warning(f"pdfplumber extraction issue: {str(e)}")
        st.warning(f"pdfplumber extraction issue: {str(e)}")
        return ""

def extract_text_with_ocr_poppler(pdf_file) -> str:
    """Extract text using OCR with Poppler and Tesseract."""
    if not POPPLER_INSTALLED:
        logger.warning("Poppler OCR extraction skipped - Poppler not installed")
        return ""
    
    start_time = time.time()
    logger.info(f"Starting OCR extraction with Poppler for {pdf_file.name}")
    try:
        # Save a copy of the PDF content
        pdf_content = pdf_file.read()
        pdf_file.seek(0)  # Reset file pointer
        
        # Convert PDF to images using Poppler
        logger.info("Converting PDF to images with Poppler")
        images = convert_from_bytes(pdf_content)
        logger.info(f"Successfully converted {len(images)} pages using Poppler")
        
        # Extract text from images using Tesseract
        text = ""
        for i, img in enumerate(images):
            logger.info(f"Processing page {i+1}/{len(images)} with Tesseract OCR")
            page_text = pytesseract.image_to_string(img)
            text += page_text + "\n"
        
        elapsed = time.time() - start_time
        words = len(text.split())
        logger.info(f"Poppler OCR extraction completed: {words} words in {elapsed:.2f} seconds")
        return text
    except Exception as e:
        logger.error(f"Poppler OCR extraction issue: {str(e)}")
        if "poppler" in str(e).lower():
            logger.error("Poppler error detected. Check PATH configuration.")
        return ""

def extract_text_with_tesseract_direct(pdf_file) -> str:
    """Extract text using Tesseract directly on the PDF."""
    if not TESSERACT_INSTALLED:
        logger.warning("Tesseract OCR extraction skipped - Tesseract not installed")
        return ""
    
    import pytesseract
    from PIL import Image
    import pdf2image
    
    start_time = time.time()
    logger.info(f"Starting direct Tesseract OCR for {pdf_file.name}")
    
    try:
        # Different approach: using Tesseract with different preprocessing
        pdf_content = pdf_file.read()
        pdf_file.seek(0)
        
        # Convert with different DPI settings
        images = convert_from_bytes(pdf_content, dpi=300)  # Higher DPI for better OCR
        
        # Extract text with different Tesseract configurations
        text = ""
        for i, img in enumerate(images):
            logger.info(f"Processing page {i+1}/{len(images)} with enhanced Tesseract settings")
            # Try with image preprocessing
            img = img.convert('L')  # Convert to grayscale
            # Apply custom Tesseract config
            page_text = pytesseract.image_to_string(
                img, 
                config='--psm 6 --oem 3'  # Page segmentation mode 6 (assume single block of text)
            )
            text += page_text + "\n"
        
        elapsed = time.time() - start_time
        words = len(text.split())
        logger.info(f"Direct Tesseract OCR completed: {words} words in {elapsed:.2f} seconds")
        return text
    except Exception as e:
        logger.error(f"Direct Tesseract OCR issue: {str(e)}")
        return ""

def extract_text_with_ocr(pdf_file) -> Tuple[str, str, str]:
    """
    Extract text using multiple OCR methods and return all results.
    Returns tuple of (combined_text, poppler_text, tesseract_text)
    """
    global POPPLER_INSTALLED, TESSERACT_INSTALLED
    
    if not (POPPLER_INSTALLED and TESSERACT_INSTALLED):
        logger.warning("OCR extraction skipped - dependencies not installed")
        return "", "", ""
    
    # Method 1: Poppler + Tesseract
    start_time = time.time()
    logger.info(f"Starting parallel OCR extraction for {pdf_file.name}")
    
    # First approach: Poppler + Tesseract
    pdf_file.seek(0)
    poppler_text = extract_text_with_ocr_poppler(pdf_file)
    poppler_words = len(poppler_text.split())
    logger.info(f"Poppler+Tesseract extraction: {poppler_words} words")
    
    # Second approach: Direct Tesseract with custom settings
    pdf_file.seek(0)
    tesseract_text = extract_text_with_tesseract_direct(pdf_file)
    tesseract_words = len(tesseract_text.split())
    logger.info(f"Enhanced Tesseract extraction: {tesseract_words} words")
    
    # Combine both texts based on which has more content
    combined_text = ""
    if poppler_words > tesseract_words:
        combined_text = poppler_text
        logger.info(f"Selected Poppler OCR result with {poppler_words} words")
    else:
        combined_text = tesseract_text
        logger.info(f"Selected Enhanced Tesseract OCR result with {tesseract_words} words")
    
    elapsed = time.time() - start_time
    logger.info(f"Combined OCR processing completed in {elapsed:.2f} seconds")
    
    return combined_text, poppler_text, tesseract_text

def extract_text_with_direct_read(pdf_file) -> str:
    """Last resort: Try to extract any text content from the PDF file directly."""
    start_time = time.time()
    logger.info(f"Starting direct text extraction for {pdf_file.name}")
    try:
        # Try to read the PDF as text directly - sometimes works for simple PDFs
        pdf_file.seek(0)
        text = ""
        content = pdf_file.read()
        
        # Try to decode as ASCII or UTF-8
        for encoding in ['utf-8', 'latin-1', 'ascii']:
            try:
                if isinstance(content, bytes):
                    text = content.decode(encoding, errors='ignore')
                elif isinstance(content, str):
                    text = content
                break
            except:
                continue
                
        # Extract anything that looks like text
        if text:
            # Remove binary data and keep only printable chars
            text = ''.join(c for c in text if c.isprintable() or c in ['\n', '\t', ' '])
            
            elapsed = time.time() - start_time
            words = len(text.split())
            logger.info(f"Direct extraction completed: {words} words in {elapsed:.2f} seconds")
            return text
            
        return ""
    except Exception as e:
        logger.warning(f"Direct extraction issue: {str(e)}")
        return ""

def clean_text(text: str) -> str:
    """Clean extracted text by removing extra whitespace and fixing common extraction issues."""
    if not text:
        return ""
    
    # Remove excessive newlines and whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove headers and footers (common in academic papers)
    text = re.sub(r'(?i)Page \d+ of \d+', '', text)
    
    # Fix common encoding issues
    text = text.replace('\u2022', '•')  # Replace Unicode bullet with ASCII
    text = text.replace('\uf0b7', '•')  # Replace Unicode bullet with ASCII
    
    # Remove any non-printable characters
    text = ''.join(c for c in text if c.isprintable() or c in ['\n', '\t', ' '])
    
    return text.strip()

def extract_text_from_pdf(pdf_files: List[tempfile._TemporaryFileWrapper]) -> Dict[str, str]:
    """Extract text content from uploaded PDF files using multiple methods."""
    pdf_contents = {}
    
    for pdf_file in pdf_files:
        try:
            # Try different extraction methods in user-specified order
            methods_tried = []
            text = ""
            ocr_details = {}
            
            # Check for cloud deployment
            is_cloud_deployment = False
            try:
                import platform
                is_cloud_deployment = platform.system() == "Linux" and os.path.exists("/home/streamlit")
            except:
                pass
                
            # Method 1: OCR first using both Poppler and Tesseract (if available)
            if OCR_AVAILABLE and POPPLER_INSTALLED and TESSERACT_INSTALLED:
                st.info(f"🔎 Trying parallel OCR extraction for {pdf_file.name} using both Poppler and Tesseract...")
                methods_tried.append("Parallel OCR")
                
                combined_text, poppler_text, tesseract_text = extract_text_with_ocr(pdf_file)
                
                # Store OCR details for display later
                ocr_details = {
                    "poppler_words": len(poppler_text.split()),
                    "tesseract_words": len(tesseract_text.split()),
                    "combined_words": len(combined_text.split()),
                    "method_selected": "Poppler" if len(poppler_text.split()) > len(tesseract_text.split()) else "Enhanced Tesseract"
                }
                
                if len(combined_text.strip()) > 100:
                    text = combined_text
                    st.success(f"✅ OCR extraction successful using {ocr_details['method_selected']} method! ({ocr_details['combined_words']} words)")
                else:
                    st.warning(f"⚠️ OCR extraction didn't yield sufficient text. Trying regular methods...")
            else:
                if is_cloud_deployment:
                    st.warning("⚠️ OCR extraction is not available in this cloud deployment. Using fallback methods.")
                else:
                    st.warning("⚠️ OCR extraction was skipped because Poppler or Tesseract is not properly installed.")
                
            # Method 2: PyPDF2 if OCR didn't yield enough text
            if len(text.strip()) < 100:
                pdf_file.seek(0)  # Reset file pointer
                methods_tried.append("PyPDF2")
                pypdf_text = extract_text_with_pypdf2(pdf_file)
                if len(pypdf_text) > len(text):
                    text = pypdf_text
            
            # Method 3: PDFPlumber if still not enough text
            if len(text.strip()) < 100:
                pdf_file.seek(0)  # Reset file pointer
                methods_tried.append("pdfplumber")
                plumber_text = extract_text_with_pdfplumber(pdf_file)
                if len(plumber_text) > len(text):
                    text = plumber_text
            
            # Method 4: Direct read as last resort
            if len(text.strip()) < 100:
                pdf_file.seek(0)  # Reset file pointer
                methods_tried.append("direct read")
                direct_text = extract_text_with_direct_read(pdf_file)
                if len(direct_text) > len(text):
                    text = direct_text
            
            # Clean the extracted text
            text = clean_text(text)
            
            # Check if we got any meaningful text
            if len(text.strip()) < 50:
                if is_cloud_deployment:
                    st.error(f"Could not extract meaningful text from {pdf_file.name}. Cloud deployment may need additional configuration for OCR capabilities.")
                    st.info("For scanned documents, consider adding `packages.txt` with OCR dependencies to your repository.")
                else:
                    st.error(f"Could not extract meaningful text from {pdf_file.name} using methods: {', '.join(methods_tried)}. The PDF might be scanned, secured, or contain non-textual content.")
                logger.error(f"Failed to extract text from {pdf_file.name} using methods: {', '.join(methods_tried)}")
                continue
                
            # Debug information
            words = len(text.split())
            st.success(f"Successfully extracted {words} words from {pdf_file.name} using methods: {', '.join(methods_tried)}")
            logger.info(f"Successfully extracted {words} words from {pdf_file.name} using methods: {', '.join(methods_tried)}")
            
            # Add OCR comparison details if available
            if ocr_details and "method_selected" in ocr_details:
                with st.expander("📊 OCR Method Comparison"):
                    st.info(f"Poppler+Tesseract extraction: {ocr_details['poppler_words']} words")
                    st.info(f"Enhanced Tesseract extraction: {ocr_details['tesseract_words']} words")
                    st.success(f"Selected method: {ocr_details['method_selected']} (better results)")
            
            pdf_contents[pdf_file.name] = text
        except Exception as e:
            st.error(f"Error processing {pdf_file.name}: {str(e)}")
            logger.error(f"Error processing {pdf_file.name}: {str(e)}", exc_info=True)
    
    return pdf_contents

def chunk_text(text, max_chunk_size=3000):
    """Split text into smaller chunks."""
    words = text.split()
    chunks = []
    current_chunk = []

    current_size = 0
    for word in words:
        word_size = len(word) + 1  # +1 for space
        if current_size + word_size > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def generate_mcqs(pdf_content: str, num_questions: int = 5, difficulty: str = "Medium") -> str:
    """Generate MCQs using the selected LLM provider API."""
    global CHUNK_SIZE
    
    # Get the active provider from session state
    active_provider = st.session_state.active_provider
    
    # Check which provider is active and get the appropriate client
    client = None
    if active_provider == "groq" and st.session_state.groq_api_key_valid:
        client = st.session_state.groq_client
    elif active_provider == "openai" and st.session_state.openai_api_key_valid:
        client = st.session_state.openai_client
    elif active_provider == "ollama" and st.session_state.ollama_host_valid:
        # Ollama doesn't use a client object like the others
        pass
    
    # For Groq and OpenAI, check if client is initialized
    if active_provider in ["groq", "openai"] and client is None:
        return "Error: API client is not initialized. Please validate your API key first."
    
    # Chunk the content if too large
    if len(pdf_content) > CHUNK_SIZE:
        chunks = chunk_text(pdf_content, max_chunk_size=CHUNK_SIZE)
        st.info(f"Document split into {len(chunks)} chunks for processing")
        
        # Process first chunk or important segments
        pdf_content = chunks[0]
        # Optionally add important parts from later chunks
        if len(chunks) > 1:
            pdf_content += "\n\nAdditional important content: " + chunks[1][:500]
    
    prompt = f"""
    Based on the following academic content, generate {num_questions} university-level multiple-choice questions (MCQs) at {difficulty} difficulty.
    
    For each question:
    1. Write a clear question based on important concepts
    2. Provide 4 options (A, B, C, D)
    3. Indicate the correct answer
    4. Give a brief explanation for why the answer is correct
    
    Content:
    {pdf_content}
    
    Format each question as:
    
    Question N: [Question text]
    A. [Option A]
    B. [Option B]
    C. [Option C]
    D. [Option D]
    
    Answer: [Correct option]
    Explanation: [Brief explanation]
    """
    
    try:
        if active_provider == "groq":
            # Use Groq client
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a brutally honest expert educator specialized in creating university-level examination questions. Your Ultimate goal is to create the most difficult questions possible from the given content/pdf files. Make sure to create questions that are not obvious and require critical thinking and understanding of the content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=2000,
            )
            return completion.choices[0].message.content
            
        elif active_provider == "openai":
            # Use OpenAI client
            completion = client.chat.completions.create(
                model="gpt-4",  # Using GPT-4 by default
                messages=[
                    {"role": "system", "content": "You are a brutally honest expert educator specialized in creating university-level examination questions. Your Ultimate goal is to create the most difficult questions possible."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=2000,
            )
            return completion.choices[0].message.content
            
        elif active_provider == "ollama":
            # Use requests to call Ollama API
            if not OLLAMA_AVAILABLE:
                return "Error: Requests package is not installed. Please install it with `pip install requests`"
                
            try:
                # Get the host URL from session state
                host_url = st.session_state.ollama_host
                
                # Prepare the request payload
                payload = {
                    "model": "llama3",  # Default model, can be made configurable
                    "messages": [
                        {"role": "system", "content": "You are a brutally honest expert educator specialized in creating university-level examination questions. Your Ultimate goal is to create the most difficult questions possible."},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.5,
                    }
                }
                
                # Make the API call
                response = requests.post(f"{host_url}/api/chat", json=payload, timeout=120)
                
                # Check if response is successful
                if response.status_code == 200:
                    result = response.json()
                    return result["message"]["content"]
                else:
                    return f"Error: Ollama API returned status code {response.status_code}"
                    
            except Exception as e:
                error_msg = str(e)
                st.error(f"Ollama API Error: {error_msg}")
                return f"Error generating MCQs with Ollama: {error_msg}"
        
        else:
            return "Error: No valid LLM provider selected. Please validate an API key first."
            
    except Exception as e:
        error_msg = str(e)
        st.error(f"API Error: {error_msg}")
        return f"Error generating MCQs: {error_msg}"

def main():
    st.title("📚 PDF to University MCQ Generator")
    
    # Initialize session state variables for all providers
    if 'groq_api_key' not in st.session_state:
        st.session_state.groq_api_key = ""
        st.session_state.groq_api_key_valid = False
        
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ""
        st.session_state.openai_api_key_valid = False
        
    if 'ollama_host' not in st.session_state:
        st.session_state.ollama_host = "http://localhost:11434"
        st.session_state.ollama_host_valid = False
        
    if 'active_provider' not in st.session_state:
        st.session_state.active_provider = "groq"  # Default provider
        
    if 'groq_client' not in st.session_state:
        st.session_state.groq_client = None
        
    if 'openai_client' not in st.session_state:
        st.session_state.openai_client = None
    
    # API Key input area at the top of the app
    with st.container():
        st.markdown("### LLM Provider Configuration")
        
        # Create tabs for different providers
        tabs = st.tabs(["Groq", "OpenAI", "Ollama"])
        
        # Groq Tab
        with tabs[0]:
            st.markdown("#### Groq API Key")
            
            # Get the API key from the user
            groq_api_key_input = st.text_input(
                "Groq API Key", 
                value=st.session_state.groq_api_key,
                type="password",
                help="Get your API key from https://console.groq.com/keys",
                key="groq_api_key_input"
            )
            
            # Save the API key to session state if it changes
            if groq_api_key_input != st.session_state.groq_api_key:
                st.session_state.groq_api_key = groq_api_key_input
                # Reset validation status when key changes
                st.session_state.groq_api_key_valid = False
                st.session_state.groq_client = None
            
            # Set up columns for the buttons
            col1, col2 = st.columns([1, 3])
            
            # Validate button in first column
            with col1:
                if st.button("Validate Groq API Key", key="validate_groq", type="primary"):
                    if validate_api_key(groq_api_key_input):
                        st.session_state.groq_api_key = groq_api_key_input
                        st.session_state.groq_api_key_valid = True
                        st.session_state.active_provider = "groq"
                        # Initialize the client and store in session state
                        st.session_state.groq_client = Groq(api_key=groq_api_key_input)
                        st.success("✅ Groq API key is valid! You can now use the app.")
                    else:
                        st.session_state.groq_api_key_valid = False
                        st.session_state.groq_client = None
                        st.error("❌ Invalid Groq API key. Please check and try again.")
            
            # Help text in second column
            with col2:
                st.markdown("""
                1. Get your API key from [Groq Console](https://console.groq.com/keys)
                2. Enter it above and click "Validate Groq API Key"
                3. Once validated, you can use the app to generate MCQs
                """)
        
        # OpenAI Tab
        with tabs[1]:
            st.markdown("#### OpenAI API Key")
            
            if not OPENAI_AVAILABLE:
                st.warning("⚠️ OpenAI package is not installed. Please install it with `pip install openai`")
            
            # Get the API key from the user
            openai_api_key_input = st.text_input(
                "OpenAI API Key", 
                value=st.session_state.openai_api_key,
                type="password",
                help="Get your API key from https://platform.openai.com/api-keys",
                key="openai_api_key_input"
            )
            
            # Save the API key to session state if it changes
            if openai_api_key_input != st.session_state.openai_api_key:
                st.session_state.openai_api_key = openai_api_key_input
                # Reset validation status when key changes
                st.session_state.openai_api_key_valid = False
                st.session_state.openai_client = None
            
            # Set up columns for the buttons
            col1, col2 = st.columns([1, 3])
            
            # Validate button in first column
            with col1:
                openai_button_disabled = not OPENAI_AVAILABLE
                if st.button("Validate OpenAI API Key", key="validate_openai", type="primary", disabled=openai_button_disabled):
                    if validate_openai_api_key(openai_api_key_input):
                        st.session_state.openai_api_key = openai_api_key_input
                        st.session_state.openai_api_key_valid = True
                        st.session_state.active_provider = "openai"
                        # Initialize the client and store in session state
                        st.session_state.openai_client = OpenAI(api_key=openai_api_key_input)
                        st.success("✅ OpenAI API key is valid! You can now use the app.")
                    else:
                        st.session_state.openai_api_key_valid = False
                        st.session_state.openai_client = None
                        st.error("❌ Invalid OpenAI API key. Please check and try again.")
            
            # Help text in second column
            with col2:
                st.markdown("""
                1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
                2. Enter it above and click "Validate OpenAI API Key"
                3. Once validated, you can use the app to generate MCQs
                """)
        
        # Ollama Tab
        with tabs[2]:
            st.markdown("#### Ollama Host Configuration")
            
            if not OLLAMA_AVAILABLE:
                st.warning("⚠️ Requests package is not installed. Please install it with `pip install requests`")
            
            # Get the host URL from the user
            ollama_host_input = st.text_input(
                "Ollama Host URL", 
                value=st.session_state.ollama_host,
                help="Enter your Ollama host URL (default: http://localhost:11434)",
                key="ollama_host_input"
            )
            
            # Save the host to session state if it changes
            if ollama_host_input != st.session_state.ollama_host:
                st.session_state.ollama_host = ollama_host_input
                # Reset validation status when host changes
                st.session_state.ollama_host_valid = False
            
            # Set up columns for the buttons
            col1, col2 = st.columns([1, 3])
            
            # Validate button in first column
            with col1:
                ollama_button_disabled = not OLLAMA_AVAILABLE
                if st.button("Validate Ollama Host", key="validate_ollama", type="primary", disabled=ollama_button_disabled):
                    if validate_ollama_host(ollama_host_input):
                        st.session_state.ollama_host = ollama_host_input
                        st.session_state.ollama_host_valid = True
                        st.session_state.active_provider = "ollama"
                        st.success("✅ Ollama host is valid and connected! You can now use the app.")
                    else:
                        st.session_state.ollama_host_valid = False
                        st.error("❌ Invalid Ollama host. Please check the URL and ensure Ollama is running.")
            
            # Help text in second column
            with col2:
                st.markdown("""
                1. Make sure Ollama is running on your local machine or remote server
                2. Enter the host URL (usually http://localhost:11434 for local installations)
                3. Click "Validate Ollama Host" to test the connection
                """)
    
    # Check if any provider is validated and active
    is_any_provider_valid = (
        (st.session_state.active_provider == "groq" and st.session_state.groq_api_key_valid) or
        (st.session_state.active_provider == "openai" and st.session_state.openai_api_key_valid) or
        (st.session_state.active_provider == "ollama" and st.session_state.ollama_host_valid)
    )
    
    if not is_any_provider_valid:
        st.warning("⚠️ Please validate at least one LLM provider to use this application")
        st.stop()  # Stop execution if no valid provider
    
    # For backward compatibility - update original api_key_valid flag
    if st.session_state.active_provider == "groq" and st.session_state.groq_api_key_valid:
        st.session_state.api_key_valid = True
    
    st.markdown("""
    Upload PDF files containing academic content, and this tool will analyze them 
    to generate university-level multiple-choice questions for exam preparation.
    """)
    
    # Display OCR status and Poppler check
    with st.expander("📋 PDF Extraction Setup Status (click to expand)", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            if POPPLER_INSTALLED:
                st.success("✅ **Poppler is installed and working!**")
            else:
                st.error("❌ **Poppler is not detected or not working properly.**")
        
        with col2:
            has_tesseract_error = False
            if len(ocr_messages) > 1:
                has_tesseract_error = "tesseract" in ocr_messages[1].lower()
            
            if has_tesseract_error:
                st.error("❌ **Tesseract OCR is not detected.**")
            elif OCR_AVAILABLE:
                st.success("✅ **Tesseract OCR is installed and working!**")
            else:
                st.error("❌ **Tesseract OCR is not configured properly.**")
        
        # Add a button to check Poppler in detail
        if st.button("🔍 Check Poppler Installation Details"):
            st.info("Checking Poppler installation details...")
            
            import subprocess
            import sys
            import os
            
            def find_poppler_in_path():
                paths = os.environ.get('PATH', '').split(os.pathsep)
                poppler_paths = []
                
                for path in paths:
                    pdftoppm_path = os.path.join(path, 'pdftoppm.exe' if sys.platform == 'win32' else 'pdftoppm')
                    if os.path.exists(pdftoppm_path):
                        poppler_paths.append(path)
                
                return poppler_paths
            
            # Try to get poppler version
            try:
                result = subprocess.run(['pdftoppm', '-v'], 
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE, 
                                        text=True, 
                                        shell=True)
                
                if result.stderr and 'version' in result.stderr.lower():
                    st.success(f"🎉 Poppler is installed! Version info: {result.stderr.strip()}")
                    st.info("Poppler commands are accessible in your PATH")
                else:
                    st.warning("Poppler seems to be installed but couldn't get version info.")
            except Exception as e:
                st.error(f"Error checking Poppler version: {str(e)}")
                st.warning("Poppler might not be in your PATH.")
            
            # Find Poppler in PATH
            poppler_paths = find_poppler_in_path()
            if poppler_paths:
                st.success(f"Found Poppler in the following PATH locations:")
                for path in poppler_paths:
                    st.code(path)
            else:
                st.warning("Couldn't find Poppler in your PATH environment variable.")
                st.info("Current PATH directories:")
                for path in os.environ.get('PATH', '').split(os.pathsep):
                    if path.strip():  # Only show non-empty paths
                        st.code(path)
        
        if not OCR_AVAILABLE:
            st.warning("""
            ⚠️ **Enhanced PDF extraction is not available.** Some PDFs (especially scanned documents) may not be processed correctly.
            """)
            
            # Check if this is likely a cloud deployment
            import platform
            if platform.system() == "Linux" and os.path.exists("/home/streamlit"):
                st.markdown("### Cloud Deployment Instructions")
                st.info("""
                It appears you're running this app on Streamlit Cloud or another cloud platform.
                
                To enable OCR functionality on Streamlit Cloud, you need to:
                
                1. Add a `packages.txt` file to your repository with these dependencies:
                ```
                poppler-utils
                tesseract-ocr
                libtesseract-dev
                ```
                
                2. Make sure your `requirements.txt` includes:
                ```
                pdf2image
                pytesseract
                ```
                
                3. Redeploying your app should install the necessary dependencies.
                """)
            else:
                st.markdown("### Installation Instructions for Windows")
                
                st.markdown("**Step 1: Install Poppler**")
                st.markdown("""
                1. Download Poppler from [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)
                2. Extract the ZIP file to a folder (e.g., `C:\\Program Files\\poppler`)
                3. Add the `bin` folder to your PATH environment variable:
                   - Right-click on 'This PC' > Properties > Advanced system settings > Environment Variables
                   - Edit the PATH variable and add the path to the bin folder (e.g., `C:\\Program Files\\poppler\\bin`)
                4. Restart your computer to ensure PATH changes take effect
                """)
                
                st.markdown("**Step 2: Install Tesseract OCR**")
                st.markdown("""
                1. Download Tesseract installer from [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki)
                2. Run the installer and select "Add to PATH" during installation
                3. Complete the installation with default options
                """)
                
                st.markdown("**Step 3: Restart the Application**")
            
            # Show specific error messages
            for msg in ocr_messages:
                st.info(msg)
            
            st.success("Once completed, restart this app for full PDF extraction capabilities")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        num_questions = st.slider("Number of MCQs per PDF", min_value=3, max_value=20, value=5)
        difficulty = st.select_slider("Difficulty Level", options=["Easy", "Medium", "Hard"], value="Medium")
        
        st.header("Advanced Settings")
        # Declare global variables first
        global MODEL, CHUNK_SIZE
        
        # Select provider first
        active_provider = st.selectbox("LLM Provider", 
                          options=["groq", "openai", "ollama"], 
                          index=0 if st.session_state.active_provider == "groq" else 
                                1 if st.session_state.active_provider == "openai" else 2,
                          help="Select the LLM provider to use")
        
        # Update the active provider in session state
        st.session_state.active_provider = active_provider
        
        # Model selection based on provider
        if active_provider == "groq":
            model = st.selectbox("Groq Model", 
                              options=["llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"], 
                              index=0,
                              help="Select the Groq model to use")
            # Update the MODEL variable based on user selection
            MODEL = model
            
            # Reinitialize client if model changes and we have a valid API key
            if st.session_state.groq_api_key_valid and st.session_state.groq_api_key:
                # Create or update the client in session state
                st.session_state.groq_client = Groq(api_key=st.session_state.groq_api_key)
        
        elif active_provider == "openai":
            model = st.selectbox("OpenAI Model", 
                              options=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"], 
                              index=0,
                              help="Select the OpenAI model to use")
            # Update the MODEL variable based on user selection
            MODEL = model
            
            # Reinitialize client if model changes and we have a valid OpenAI key
            if st.session_state.openai_api_key_valid and st.session_state.openai_api_key:
                # Create or update the client in session state
                st.session_state.openai_client = OpenAI(api_key=st.session_state.openai_api_key)
        
        elif active_provider == "ollama":
            model = st.selectbox("Ollama Model", 
                              options=["llama3", "llama3:8b", "mistral", "llama2", "phi3"], 
                              index=0,
                              help="Select the Ollama model to use")
            # Update the MODEL variable based on user selection
            MODEL = model
        
        # Add a "Test Selected Provider" button
        if st.button("Test Selected Provider"):
            # Check which provider is selected and validate it
            if active_provider == "groq":
                if st.session_state.groq_api_key_valid:
                    st.success("✅ Groq API key is valid and ready to use!")
                elif st.session_state.groq_api_key:
                    # Try to validate the key
                    if validate_api_key(st.session_state.groq_api_key):
                        st.session_state.groq_api_key_valid = True
                        st.success("✅ Groq API key is valid!")
                        # Initialize the client
                        st.session_state.groq_client = Groq(api_key=st.session_state.groq_api_key)
                    else:
                        st.error("❌ Invalid Groq API key. Please update it in the configuration tab.")
                else:
                    st.warning("⚠️ No Groq API key provided. Please enter it in the configuration tab.")
                    
            elif active_provider == "openai":
                if st.session_state.openai_api_key_valid:
                    st.success("✅ OpenAI API key is valid and ready to use!")
                elif st.session_state.openai_api_key:
                    # Try to validate the key
                    if validate_openai_api_key(st.session_state.openai_api_key):
                        st.session_state.openai_api_key_valid = True
                        st.success("✅ OpenAI API key is valid!")
                        # Initialize the client
                        st.session_state.openai_client = OpenAI(api_key=st.session_state.openai_api_key)
                    else:
                        st.error("❌ Invalid OpenAI API key. Please update it in the configuration tab.")
                else:
                    st.warning("⚠️ No OpenAI API key provided. Please enter it in the configuration tab.")
                    
            elif active_provider == "ollama":
                if st.session_state.ollama_host_valid:
                    st.success("✅ Ollama host is valid and connected!")
                elif st.session_state.ollama_host:
                    # Try to validate the host
                    if validate_ollama_host(st.session_state.ollama_host):
                        st.session_state.ollama_host_valid = True
                        st.success("✅ Ollama host is valid and connected!")
                    else:
                        st.error("❌ Invalid Ollama host. Please ensure Ollama is running at the specified host.")
                else:
                    st.warning("⚠️ No Ollama host provided. Please enter it in the configuration tab.")
        
        with st.expander("Chunking Settings"):
            st.info("For large PDFs, the content will be split into smaller chunks")
            chunk_size = st.slider("Max Chunk Size (words)", 
                                min_value=1000, 
                                max_value=8000, 
                                value=4000,
                                help="Larger values may cause API errors for rate limits")
            CHUNK_SIZE = chunk_size
    
    # File uploader for multiple PDFs
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type="pdf", 
        accept_multiple_files=True,
        help="Upload academic PDFs containing the content for which you want to generate MCQs."
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} file(s) uploaded")
        
        # Process button
        if st.button("Generate MCQs", type="primary"):
            with st.spinner("Processing PDFs and generating MCQs..."):
                # Extract text from PDFs
                pdf_contents = extract_text_from_pdf(uploaded_files)
                
                # Display results in tabs
                if pdf_contents:
                    processed_files = list(pdf_contents.keys())
                    tabs = st.tabs([f"{pdf_name}" for pdf_name in processed_files])
                    
                    for i, (pdf_name, content) in enumerate(pdf_contents.items()):
                        with tabs[i]:
                            st.subheader(f"Processing {pdf_name}")
                            
                            # Show content statistics
                            words = len(content.split())
                            st.info(f"📊 Extracted {words:,} words from PDF")
                            
                            # For debugging: show a sample of the extracted text
                            with st.expander("Preview extracted content"):
                                st.text(content[:1000] + "..." if len(content) > 1000 else content)
                            
                            # Generate MCQs
                            if words >= 50:  # Lower threshold for valid content
                                with st.spinner(f"Generating MCQs for {pdf_name}..."):
                                    mcqs = generate_mcqs(content, num_questions, difficulty)
                                
                                if not mcqs.startswith("Error"):
                                    st.subheader("📝 Generated MCQs")
                                    st.markdown(mcqs)
                                    
                                    # Add download button for MCQs
                                    st.download_button(
                                        label="⬇️ Download MCQs",
                                        data=mcqs,
                                        file_name=f"MCQs_{pdf_name.replace('.pdf', '')}.txt",
                                        mime="text/plain"
                                    )
                                else:
                                    st.error(mcqs)
                            else:
                                st.warning(f"⚠️ The content extracted from {pdf_name} is too short to generate meaningful MCQs.")
                else:
                    st.error("❌ Failed to extract content from any of the uploaded PDFs. Please try different PDF files.")
    else:
        st.info("Please upload PDF files to get started.")
        
        # Example section
        with st.expander("See Example"):
            st.markdown("""
            **Example MCQs:**
            
            **Question 1:** Which data structure operates on a Last-In-First-Out (LIFO) principle?
            A. Queue
            B. Stack
            C. Linked List
            D. Binary Tree
            
            **Answer:** B. Stack
            **Explanation:** A stack follows the LIFO principle where the last element inserted is the first one to be removed.
            """)
    
    # Add attribution footer
    st.markdown("---")
    st.markdown(
        """<div style='text-align: center; color: gray; font-size: 0.8em;'>
        Created by <a href='https://github.com/Ishaan-Rai09' target='_blank'>Ishaan Rai</a> | 
        © 2024
        </div>""", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 