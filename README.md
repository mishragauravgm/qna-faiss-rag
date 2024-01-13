# qna-faiss-rag

```markdown
# Document Querying using RAG, FAISS and LangChain

## Overview

This project utilizes OpenAI's language models, Langchain, and Faiss for semantic search to create a custom summarization app. The app is designed to answer questions from any PDF document through a combination of semantic search and retrieval augmented generation (RAG).

## Dependencies

Ensure you have the necessary dependencies installed by running:

```bash
pip install -r requirements.txt
```

The key dependencies include:
- [OpenAI](https://github.com/openai/openai-python): Python wrapper for OpenAI API
- [PyPDF2](https://pythonhosted.org/PyPDF2/): Library for reading PDF files
- [Langchain](https://github.com/langchain/langchain): A library for natural language processing tasks
- [Streamlit](https://streamlit.io/): A web app framework for creating interactive data applications

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/mishragauravgm/qna-faiss-rag
cd qna-faiss-rag
```

2. Set up your environment variables by creating a `.env` file with your OpenAI API key:

```bash
OPENAI_API_KEY=your-api-key
```

3. Run the application:

```bash
streamlit run app.py
```

## Usage

1. Enter a user prompt and the path to a PDF file in the provided input fields.
2. Click the "Generate Answer" button to obtain the answer to your question.

## Project Structure

The main components of the project include:

- **app.py:** The Streamlit web application.
- **langchain:** A library for natural language processing tasks.
- **requirements.txt:** List of Python dependencies for the project.

## Additional Information

- **RAG (Retrieval Augmented Generation):** Utilizes Faiss for semantic search and the Hugging Face model repository for language models.
- **Faiss:** A library for efficient similarity search and clustering of dense vectors.

Feel free to explore and customize the code for your specific use case.
```

Make sure to replace "your-username" and "your-repo" with your GitHub username and the repository name, respectively. Adjust the project structure section based on the actual structure of your project.