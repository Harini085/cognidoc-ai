# Cognidoc AI — PDF Question Answering System

Cognidoc AI is a simple AI-powered application that allows users to upload PDF documents and ask questions about their content. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant passages and generate answers using a Large Language Model (LLM).

## Features

- **PDF Upload:** Upload PDF documents for analysis.
- **Text Extraction:** Extract text from PDF pages using PyMuPDF.
- **Text Chunking:** Divide document text into smaller, manageable chunks.
- **Semantic Search:** Generate embeddings and search for relevant passages using FAISS.
- **AI-Powered Answers:** Generate contextual answers using the Groq API.
- **Source Passages:** Display relevant document passages to support generated answers.
- **Interactive Interface:** Ask questions through a Streamlit chat interface.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | User interface |
| PyMuPDF | PDF text extraction |
| Sentence Transformers | Text embeddings |
| FAISS | Vector similarity search |
| Groq API | LLM-based answer generation |
| Pytest | Automated testing |

## How It Works

1. Upload a PDF document.
2. Extract text from the document.
3. Split the extracted text into chunks.
4. Convert the chunks into vector embeddings.
5. Store and search the embeddings using FAISS.
6. Retrieve the most relevant passages for the user's question.
7. Generate an answer using the Groq API and display supporting passages.

## Project Structure

```text
cognidoc-ai/
├── app/
│   ├── main.py
│   ├── pdf_processor.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── answer_generator.py
│   ├── test_pdf_processor.py
│   ├── test_chunker.py
│   ├── test_embeddings.py
│   ├── test_retriever.py
│   ├── test_answer_generator.py
│   ├── test_pipeline.py
│   └── test_answer_quality.py
├── data/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Harini085/cognidoc-ai.git
cd cognidoc-ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root and add your Groq API credentials:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Obtain an API key from the [Groq Console](https://console.groq.com/keys).

**Security:** Never commit your `.env` file or expose your API key publicly.

### 5. Run the application

```bash
python -m streamlit run app/main.py --server.fileWatcherType none
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## Running Tests

Run the automated test suite from the project root:

```bash
python -m pytest app/ -v
```

The project's seven automated tests have passed in the development environment.

## Deployment

Cognidoc AI can be deployed using [Streamlit Community Cloud](https://share.streamlit.io/).

1. Connect your GitHub repository.
2. Set the main file path to `app/main.py`.
3. Configure `GROQ_API_KEY` and `GROQ_MODEL` under the app's Secrets settings.
4. Deploy the application.

## Limitations

- Answer quality depends on the PDF's text quality and the relevance of retrieved passages.
- AI-generated answers may contain inaccuracies and should be checked against the original document.
- A valid Groq API key and internet connection are required for AI answer generation.

## Future Improvements

- Support for multiple PDF documents.
- Improved retrieval and answer evaluation.
- Conversation history and document management.

## Author

**Harini S**

M.Sc. Data Science

GitHub: [Harini085](https://github.com/Harini085)

LinkedIn: [harini0805](https://www.linkedin.com/in/harini0805)

Portfolio: [harini085.github.io](https://harini085.github.io/)

---

*Built with Python, semantic search, and Retrieval-Augmented Generation (RAG).* 
