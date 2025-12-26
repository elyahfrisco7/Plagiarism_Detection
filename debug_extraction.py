from pdf2image import convert_from_path
import os
import sys

# Use one of the found PDFs
pdf_path = r"F:\RECHERCHE\THESE\project\uploads\Memoire_Fanny.pdf"

if not os.path.exists(pdf_path):
    print(f"File not found: {pdf_path}")
    sys.exit(1)

print(f"Testing extraction on {pdf_path}")
try:
    images = convert_from_path(pdf_path)
    print(f"Success! Extracted {len(images)} images.")
except Exception as e:
    print(f"Error: {e}")
