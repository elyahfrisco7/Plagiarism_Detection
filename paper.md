---
title: 'An Application for Detecting Plagiarism in University Theses'
tags:
  - Python
  - Flask
  - plagiarism detection
  - NLP
  - computer vision
  - academic integrity
authors:
  - name: Elyah Frisco Andriantsialo
    affiliation: 1
  - name: Volatiana Marielle Ratianantitra
    affiliation: 1
  - name: Thomas Mahatody
    affiliation: 1
affiliations:
 - name: Laboratory for Mathematical and Computer Applied to the Development Systems, University of Fianarantsoa, Madagascar
   index: 1
date: 26 December 2025
bibliography: paper.bib
---

# Summary

Academic plagiarism has evolved beyond simple copy-paste text to include complex paraphrasing and the reuse of visual elements like figures and diagrams. To address this, we present a hybrid, multimodal web application designed for contextualized plagiarism detection. The system utilizes a multi-criteria approach, analyzing documents based on six distinct dimensions: Theme, Location, Methodology, Results, Global Content, and Images (THLME-Gre schema).

Built with **Flask**, the application leverages advanced semantic models—specifically **Sentence-BERT** [@Reimers:2019] for textual analysis and **CLIP** (Contrastive Language-Image Pre-training) [@Radford:2021] for visual analysis. It employs a vector database (ChromaDB) to perform efficient Approximate Nearest Neighbor (ANN) searches across large repositories of university theses.

![Screenshot of the application interface showing the dashboard and analysis results.](Application.png)

# Statement of Need

Ensuring academic integrity is a growing challenge for higher education institutions in Madagascar, particularly at the **University of Fianarantsoa**, which manages over 30,000 students across various doctoral schools and departments. Currently, the university lacks a centralized, automated institutional tool for plagiarism detection. Faculty members often rely on manual verification or commercial tools that primarily focus on English content and surface-level text matching.

These existing solutions present two major limitations for our context:
1.  **Language and Context:** The majority of student theses are written in **French**. Generic tools often struggle to distinguish between legitimate thematic overlap (e.g., multiple students working on "Web Design" or "Digitalization") and actual plagiarism. Our system addresses this by explicitly modeling the "Study Location" and "Methodology" as separate semantic criteria, reducing false positives caused by common academic jargon or shared internship locations.
2.  **Multimodality:** Traditional tools frequently miss "visual plagiarism," where students might rewrite the text but copy diagrams, charts, or results directly. By integrating CLIP, our application detects similarities in visual content that text-only tools overlook [@Chowdhury:2016].

This software provides a robust, scalable, and locally deployable solution to enforce academic honesty, specifically tailored to the linguistic and structural needs of Malagasy university research.

# Implementation and Architecture

The application follows a modular architecture. The core processing pipeline handles PDF extraction, separating text and images. 
- **Text** is encoded into dense vectors using `SentenceTransformer` to capture deep semantic meaning [@Devlin:2019].
- **Images** are processed via `CLIP` to project visual data into a shared embedding space.
- **Data Storage** is hybrid: metadata and structured criteria (Theme, Location, etc.) are stored in a Relational DBMS, while high-dimensional embeddings are indexed in a Vector Database for real-time retrieval.

![System Architecture: Data flow from PDF extraction to hybrid storage (Relational and Vector Database).](Architecture.png)

The global similarity score ($S_{global}$) is computed using an egalitarian weighting model, aggregating cosine similarities from the six defined criteria. This allows for a nuanced assessment, providing decision support thresholds (e.g., >80% for high suspicion) rather than a simple binary judgment.


# References