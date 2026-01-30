import sys
print(f"Python version: {sys.version}")

try:
    import flask
    print("flask imported successfully")
except ImportError as e:
    print(f"Error importing flask: {e}")

try:
    import sqlalchemy
    print("sqlalchemy imported successfully")
except ImportError as e:
    print(f"Error importing sqlalchemy: {e}")

try:
    import pymysql
    print("pymysql imported successfully")
except ImportError as e:
    print(f"Error importing pymysql: {e}")

try:
    import PyPDF2
    print("PyPDF2 imported successfully")
except ImportError as e:
    print(f"Error importing PyPDF2: {e}")

try:
    import sentence_transformers
    print("sentence_transformers imported successfully")
except ImportError as e:
    print(f"Error importing sentence_transformers: {e}")

try:
    import chromadb
    print("chromadb imported successfully")
except ImportError as e:
    print(f"Error importing chromadb: {e}")

try:
    import werkzeug
    print("werkzeug imported successfully")
except ImportError as e:
    print(f"Error importing werkzeug: {e}")

try:
    import PIL
    print("PIL imported successfully")
except ImportError as e:
    print(f"Error importing PIL: {e}")

try:
    import torch
    print("torch imported successfully")
except ImportError as e:
    print(f"Error importing torch: {e}")

try:
    import ftfy
    print("ftfy imported successfully")
except ImportError as e:
    print(f"Error importing ftfy: {e}")

try:
    import regex
    print("regex imported successfully")
except ImportError as e:
    print(f"Error importing regex: {e}")

try:
    import tqdm
    print("tqdm imported successfully")
except ImportError as e:
    print(f"Error importing tqdm: {e}")

try:
    import fitz
    print("fitz (pymupdf) imported successfully")
except ImportError as e:
    print(f"Error importing fitz: {e}")

try:
    import numpy
    print("numpy imported successfully")
except ImportError as e:
    print(f"Error importing numpy: {e}")

try:
    import clip
    print("clip imported successfully")
except ImportError as e:
    print(f"Error importing clip: {e}")
