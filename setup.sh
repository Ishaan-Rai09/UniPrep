#!/bin/bash

# Echo versions
echo "Checking installed dependencies:"
echo "-------------------------------"

# Check Tesseract
if command -v tesseract &> /dev/null
then
    echo "✅ Tesseract is installed:"
    tesseract --version | head -n 1
else
    echo "❌ Tesseract is NOT installed"
fi

# Check Poppler
if command -v pdftoppm &> /dev/null
then
    echo "✅ Poppler is installed:"
    pdftoppm -v 2>&1 | head -n 1
else
    echo "❌ Poppler is NOT installed"
fi

echo "-------------------------------"
echo "Setup complete." 