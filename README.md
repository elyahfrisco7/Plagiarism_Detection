# 🎓 ThesisPlag: Academic Thesis Plagiarism Detection

ThesisPlag is a Flask-based application for detecting academic plagiarism between theses by combining semantic similarity, document content analysis, and visual comparison. It uses Sentence-BERT for text embeddings, CLIP for image embeddings, and ChromaDB for semantic indexing.

The project stores thesis metadata in MySQL and exposes a web interface for uploading PDFs, searching similar documents, comparing two theses, and scanning the database for the most relevant matches.

## Key features

- PDF upload and metadata capture for academic theses
- Text extraction from PDF using PyMuPDF/PyPDF2
- Image extraction from PDF pages and visual similarity using OpenAI CLIP
- Semantic search over thesis content with ChromaDB
- Weighted comparison across six criteria:
  - Theme
  - Stage/location
  - Methodology
  - Results
  - Full content
  - Images
- Automatic metadata inference from raw extracted PDF text using Anthropic Claude
- Web UI and JSON APIs for comparison and scan workflows
- Configuration for criteria weights via the /settings page

## Technology stack

- Python 3.8+
- Flask
- SQLAlchemy
- MySQL
- ChromaDB
- sentence-transformers
- OpenAI CLIP
- PyMuPDF
- PyPDF2
- Pillow
- NumPy
- Anthropic API
- HTML / CSS / JavaScript

## Project structure

```text
.
├── app/
│   ├── __init__.py
│   ├── ai_extractor.py
│   ├── config.py
│   ├── controllers.py
│   ├── models.py
│   ├── routes.py
│   ├── utils.py
│   ├── weights.py
│   └── __pycache__/
├── static/
├── templates/
├── tests/
│   ├── conftest.py
│   ├── test_controllers.py
│   ├── test_models.py
│   ├── test_routes.py
│   └── test_utils.py
├── .gitignore
├── .env.example (optional, not committed)
├── LICENSE
├── README.md
├── requirements.txt
├── run.py
├── paper.md
├── paper.pdf
├── paper.bib
├── verify_imports.py
├── uploads/
├── chroma_db/
├── weights_config.json   # created at runtime if weights are customized
└── .github/
```

## Prerequisites

Before running the project, make sure you have:

- Python 3.8 or newer
- MySQL Server
- pip and virtualenv support
- Git
- A working internet connection to install dependencies and the CLIP package from GitHub
- An Anthropic API key if you want to use the AI metadata extraction feature

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/elyahfrisco7/Plagiarism_Detection.git
cd Plagiarism_Detection
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

On Linux/macOS:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install OpenAI CLIP

The project depends on the official OpenAI CLIP repository:

```bash
pip install git+https://github.com/openai/CLIP.git
```

## Environment configuration

Create a `.env` file in the project root if you want to use the AI metadata extraction feature:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

The application loads this file via `python-dotenv` in `app/config.py`.

## Database setup

The project uses MySQL with SQLAlchemy. Create the database and matching table before running the app.

```sql
CREATE DATABASE IF NOT EXISTS thesis_db;
USE thesis_db;

CREATE TABLE IF NOT EXISTS `theses` (
    `id` INT AUTO_INCREMENT,
    `title` VARCHAR(255),
    `theme` VARCHAR(255),
    `author` VARCHAR(255),
    `university` VARCHAR(255),
    `thesis_type` VARCHAR(50),
    `stage_location` VARCHAR(255),
    `methodology` TEXT,
    `results` TEXT,
    `pdf_path` VARCHAR(255),
    `theme_embedding` TEXT,
    `stage_embedding` TEXT,
    `methodology_embedding` TEXT,
    `results_embedding` TEXT,
    `content_embedding` TEXT,
    `images_embedding` TEXT,
    PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

Then configure the database URI in `app/config.py`:

```python
DATABASE_URI = 'mysql+pymysql://username:password@127.0.0.1:3306/thesis_db'
```

The default configuration in the repository is:

```python
DATABASE_URI = 'mysql+pymysql://root:@127.0.0.1:3306/thesis_db'
```

Update it to your local MySQL credentials before starting the application.

## Runtime behavior

The app creates these runtime directories/files automatically when needed:

- `uploads/` for uploaded PDFs
- `chroma_db/` for ChromaDB persistence
- `weights_config.json` for saved comparison weights

## Running the application

From the project root:

```bash
python run.py
```

Then open:

```text
http://127.0.0.1:5000
```

## How the app works

### Upload workflow

1. User uploads a thesis PDF from the homepage.
2. The app extracts text and images from the PDF.
3. It generates text embeddings with SentenceTransformer.
4. It generates a visual embedding with CLIP.
5. It stores the thesis metadata and embeddings in MySQL.
6. It indexes the full document content in ChromaDB for semantic retrieval.

### Comparison workflow

- The app compares two thesis records using cosine similarity.
- Each of the six criteria is scored separately and then combined using configurable weights.
- The overall similarity score is returned in percent format.

### Search workflow

- A user enters a natural language query.
- The query is embedded with the same SentenceTransformer model.
- ChromaDB retrieves the most similar theses.
- The results are displayed in the interface.

### AI metadata extraction

The `/api/extract-metadata` endpoint extracts raw text from the PDF and sends it to Claude Haiku for structured metadata inference. This is used to auto-fill the upload form and reduce manual entry.

## Routes and APIs

The Flask application includes these main routes:

- `/` – homepage
- `/upload` – upload a thesis PDF and metadata
- `/search` – semantic search over indexed theses
- `/compare` – compare two theses by ID
- `/theses` – list all theses
- `/settings` – edit comparison weighting
- `/api/extract-metadata` – auto-fill thesis fields from a PDF
- `/api/scan` – scan all theses against a selected one
- `/api/theses/<thesis_id>` – delete a thesis and associated index data

## Testing

Run the test suite with:

```bash
pytest tests/
```

The repository includes tests for routes, controllers, models, and utility behavior.

## Notes

- `app/config.py` creates a singleton SentenceTransformer model, CLIP model, and ChromaDB client.
- The weights used in the similarity breakdown are stored in `weights_config.json` and normalized automatically.
- The upload folder is in the project root and is not intended as permanent storage for production deployments unless configured appropriately.
- The project is designed for academic/documentation use and is not a production-grade enterprise service by default.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation / research context

This codebase is designed for multimodal plagiarism detection using the THLME-Gre schema and combines textual and visual similarity for academic theses.

If you want, I can also help with one of the following:

- add a proper `.env.example` file
- improve the README with screenshots and usage examples
- fix runtime issues in the codebase
- prepare a Docker setup for easier deployment

