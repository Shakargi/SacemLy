# Sacemly – Intelligent Learning System for Independent Students

## Table of Contents

* [Overview](#overview)
* [Project Purpose](#project-purpose)
* [Key Features](#key-features)
* [Target Audience](#target-audience)
* [System Architecture](#system-architecture)
* [Installation and Usage](#installation-and-usage)
* [Technologies Used](#technologies-used)
* [Development Status](#development-status)
* [License](#license)
* [Contact Information](#contact-information)

---

## Overview

**Sacemly** is an AI-powered learning support system designed to assist students and independent learners by transforming educational content into structured, efficient, and personalized study tools. While intelligent summarization remains a core feature, Sacemly aims to be a complete self-learning assistant by offering a comprehensive suite of tools for academic understanding and mastery.

The project is built entirely from the ground up, without relying on pre-trained language models. All core algorithms—such as word embeddings, context modeling, and importance ranking—are implemented manually to ensure a deep understanding of the underlying mechanisms.

---

## Project Purpose

The primary goal of Sacemly is to empower learners to take full control of their educational journey through smart automation and personalization. It aims to:

* Provide all essential tools for effective self-learning.
* Reduce cognitive load during reading and review.
* Deliver summaries and structured insights in the user's personal writing style.
* Highlight the most critical information within content.
* Enable more effective and customized study workflows.

This project is part of a larger initiative to develop AI solutions that are transparent, customizable, and educational by design.

---

## Key Features

* **Comprehensive Self-Learning Toolkit**: Beyond summarization, Sacemly provides contextual insights, style mimicry, and study-friendly formatting.
* **Personalized Summarization**: The system adapts to the user’s writing patterns to produce summaries in a familiar and natural tone.
* **Importance-Weighted Analysis**: Each part of the text is evaluated and scored based on its semantic importance, ensuring that essential ideas are emphasized.
* **Style Preservation**: The system preserves the user’s voice, vocabulary, and phrasing throughout the summarization process.
* **Document and PDF Support**: Input can be provided through plain text, PDF files, or structured educational materials.
* **Social Content Integration**: The system supports the import of educational content from online posts, forums, or transcripts.
* **Privacy-First Design**: All data remains local or within a user-controlled cloud environment, ensuring full data privacy.
* **Adaptive Learning System**: Sacemly improves over time by analyzing user feedback and past summaries.

---

## Target Audience

Sacemly is designed for:

* University students seeking efficient study tools.
* High school students preparing for exams or assignments.
* Independent learners managing large volumes of reading material.
* Educators and instructors preparing learning materials.
* Individuals with attention difficulties or reading fatigue.

---

## System Architecture

The system operates in the following stages:

1. **Text Input**: The user provides content via file upload or direct text input.
2. **Preprocessing**: Text is cleaned, segmented, and normalized for analysis.
3. **Style Learning**: The system identifies unique stylistic patterns from the user's existing content.
4. **Importance Detection**: Using manually implemented NLP algorithms, the system scores and ranks textual segments.
5. **Summary Generation and Study Support**: A custom summary is generated alongside suggestions and tools to enhance self-study.

---

## Installation and Usage

> Note: The system is currently under development. Instructions below apply to the initial functional prototype.

### Clone the Repository

```bash
git clone https://github.com/Shakargi/Sacemly.git
cd Sacemly
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python main.py
```

---

## Technologies Used

* **Programming Language**: Python 3.x
* **Core Libraries**: NumPy, pandas (for mathematical and structural operations)
* **Natural Language Processing**: Custom implementations of Word2Vec, CBOW, and importance-weighted scoring
* **Document Processing**: PDF parsing and text extraction tools
* **Interface (Planned)**: React-based or Streamlit web interface

---

## Development Status

As of 2025, Sacemly is in active development. The following milestones have been achieved or are in progress:

* [x] Completion of theoretical and algorithmic groundwork
* [x] Manual implementation of Word2Vec and CBOW
* [x] Importance analysis and content ranking
* [x] Style recognition and imitation module (currently under development)
* [ ] PDF and social content ingestion
* [ ] Full user interface and frontend integration

---

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute the code, provided that proper credit is given to the original author.

---

## Contact Information

For inquiries, collaboration opportunities, or additional information, please contact:

**Email**: [avrahamsh124@gmail.com](mailto:avrahamsh124@gmail.com)
**Project Website (planned)**: [https://sacemly.ai](https://sacemly.ai)

---
