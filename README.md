# 🎓 ThesisPlag: Academic Thesis Plagiarism Detection

ThesisPlag is a powerful Flask-based application designed to detect similarities between academic theses using advanced NLP (SentenceTransformers) and Computer Vision (CLIP) models. It performs multi-criteria analysis including theme, methodology, results, and even image similarity.

---

## Features

- **Multi-Criteria Comparison**: Compare theses based on theme, location, methodology, results, and full content.
- **Visual Similarity**: Extract and compare images from PDFs using OpenAI's CLIP model.
- **Semantic Search**: Find relevant theses using natural language queries powered by ChromaDB.
- **MVC Architecture**: Clean and maintainable code structure.
- **Modern UI**: Sleek, dark-themed interface for a premium user experience.

---

## 🛠️ Prerequisites

- **Python 3.8+**
- **MySQL Server**
- **Git** (for CLIP installation)

---

##  Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd thesis-plagiarism-detection
```

### 2. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.
```bash
python -m venv venv
```

### 3. Activate the Virtual Environment
- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Install OpenAI CLIP
CLIP must be installed directly from the official GitHub repository:
```bash
pip install git+https://github.com/openai/CLIP.git
```

---

##  Database Setup

### 1. Create Database and Table
Run the following SQL script in your MySQL client (e.g., phpMyAdmin, MySQL Workbench):

```sql
CREATE DATABASE IF NOT EXISTS thesis_db;
USE thesis_db;

CREATE TABLE IF NOT EXISTS `theses` (
    `id` INT AUTO_INCREMENT,
    `title` VARCHAR(255),
    `theme` VARCHAR(255),
    `author` VARCHAR(255),
    `university` VARCHAR(255),
    `thesis_type` VARCHAR(50),      -- "research" or "professional"
    `stage_location` VARCHAR(255),  -- Internship or study location
    `methodology` TEXT,             -- Methodology and objectives
    `results` TEXT,                 -- Results (technologies, tools, etc.)
    `pdf_path` VARCHAR(255),        -- PDF filename
    `theme_embedding` TEXT,         -- Theme embedding (JSON)
    `stage_embedding` TEXT,         -- Location embedding (JSON)
    `methodology_embedding` TEXT,   -- Methodology embedding (JSON)
    `results_embedding` TEXT,       -- Results embedding (JSON)
    `content_embedding` TEXT,       -- Full content embedding
    `images_embedding` TEXT,        -- Average image embedding
    PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### 2. Configure Connection
Open `app/config.py` and update the `DATABASE_URI` with your MySQL credentials:
```python
DATABASE_URI = 'mysql+pymysql://username:password@127.0.0.1:3306/thesis_db'
```

---

##  Running the Application

Start the Flask development server:
```bash
python run.py
```
The application will be available at `http://127.0.0.1:5000`.

---

##  Usage

- **Upload**: Use the home page to upload a new thesis PDF and fill in the metadata.
- **Search**: Enter a natural language query in the search bar to find similar theses.
- **Compare**: Select two theses from the list (`/theses`) to see a detailed similarity breakdown.
- **Scan**: Use the "Scan" feature on a thesis to automatically find the most similar documents in the database.

---

##  Running Tests

To run the automated unit tests:
```bash
pytest tests/
```

---

## 📄 License
This project is for academic research purposes.

