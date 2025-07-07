# AI-ML-Code-Projects
A collection of AI and ML code projects, including model implementations, regression techniques, classification tasks, and data pipeline setup. This repository is aimed at showcasing hands-on work in AI/ML development and deployment.
# 🎯 Interview Flash Assistant

Quick-access Q&A app for AI Engineer interview prep, built in React.

## 🔥 Live Demo

[Click here to use my Interview Flash Assistant](https://kjnlyp.csb.app)

# AskMyDocs-RAG: Retrieval Augmented Generation Pipeline

**Project Name:** AskMyDocs-RAG
**Primary Developer:** Loretta Gray
**Platform:** Google Colab & Jupyter Notebook
**Target Audience:** Dovetail Interview Team, Future Clients, Contract Developers
**GitHub:** [Loretta991](https://github.com/Loretta991)
**Slack Channel:** #ellegreyllc-rag-developer

---

## 🔧 Project Overview

AskMyDocs-RAG demonstrates a fully functional Retrieval-Augmented Generation (RAG) pipeline for document-based Q\&A, using OpenAI's GPT models and FAISS vector search.

The project emphasizes modular design, clear documentation, and robust fallback logic, allowing:

* Teams to understand each RAG component step-by-step
* Development and testing even without active API tokens (Mock Mode)
* Contractors to follow structured, repeatable workflows

This project is designed to be shared via GitHub and executed in Google Colab for ease of access and testing.

---

## 📦 Requirements

```
openai
faiss-cpu
sentence-transformers
numpy
PyPDF2 (optional for PDF support)
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🗂️ File Structure

```
askmydocs-rag/
├── requirements.txt
├── rag_pipeline.ipynb           # Main Colab notebook
├── my_faiss.index               # FAISS vector index
├── chunks.pkl                   # Serialized text chunks
├── openai-key.txt (excluded)    # Local-only API key file
```

**Note:** Do not upload `openai-key.txt` to GitHub. Keep API keys secure and local.

---

## 🚀 How to Run the RAG Pipeline (Colab)

1. Clone or download this project to your machine.
2. Open `rag_pipeline.ipynb` in Google Colab.
3. Upload the following:

   * `openai-key.txt` containing your OpenAI API key
   * `my_faiss.index` (pre-built FAISS index)
   * `chunks.pkl` (text chunks for retrieval)
4. Execute notebook cells in order:

   * **Step 1:** Install libraries (if required)
   * **Step 2:** Load your document(s)
   * **Step 3:** Chunk documents for vector storage
   * **Step 4:** Build or load FAISS index
   * **Step 5:** Load API key with Mock Mode fallback
   * **Step 6:** Run retrieval and generation tests

---

## 🧠 Mock Mode: Token-Free Testing

If no API token is present, invalid, or quota exceeded, the project auto-enables Mock Mode, simulating GPT responses to keep development flowing.

Manual override example:

```python
USE_MOCK_MODE = True
```

Mock response output:

```
[MOCK GPT CALL] Using context:
...
(Simulated GPT response to your question)
```

This feature ensures that developers can demonstrate functionality and pipeline structure without needing paid API tokens.

---

## 🎯 Project Purpose

AskMyDocs-RAG provides:

* Working Retrieval-Augmented Generation pipeline
* Logical, annotated development process
* Robust fallback for testing without tokens
* Template for scalable document-based assistants or chatbots
* A "User's Manual" style guide for future contractors or team members
* Ready-to-upload GitHub project designed to help contractors onboard quickly

---

## 💡 Future Enhancements

* Multi-format document loaders (PDF, DOCX)
* Smarter chunking strategies
* Streamlit or lightweight web front-end
* API-ready deployment structure
* Expanded README with visual examples or Colab demo links

---

## 📬 Contact & Support

* GitHub: [Loretta991](https://github.com/Loretta991)
* Slack: #ellegreyllc-rag-developer
* Email: [ellegrey@ellegreyllc.com](mailto:ellegrey@ellegreyllc.com)

---

**"Tested. Mocked. Structured for Scale. Ready for GitHub."*

## 📄 Resume

[Download My Resume (PDF)](https://github.com/Loretta991/AI-ML-Code-Projects/blob/main/Loretta-Gray-Resume.pdf)

