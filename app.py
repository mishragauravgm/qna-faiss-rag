from openai import OpenAI
import os
from dotenv import load_dotenv
import PyPDF2 as pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

from langchain_community.llms import HuggingFaceHub
import streamlit as st



load_dotenv()

#os.chdir('/content/drive/MyDrive/Colab Notebooks/LangchainRAGQA')
client = OpenAI(api_key = os.environ.get('OPENAI_API_KEY'))


def pdf_to_faiss(pdf_location, chunk_size=800, chunk_overlap=100):
    #pdf_location = '/content/drive/MyDrive/Colab Notebooks/LangchainRAGQA/budget_speech.pdf'
    pdf = pypdf.PdfReader(pdf_location)
    full_text = ''
    for i, content in enumerate(pdf.pages):
        raw_text = content.extract_text()
        full_text += raw_text
    text_splits = RecursiveCharacterTextSplitter(separators='\n', chunk_size = 800, chunk_overlap = 100, length_function=len).split_text(full_text)
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_texts(text_splits, embeddings)
    return db


docs=''
def answer(db, question):
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1" #"google/flan-t5-base"
    model = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.7, "max_length": 5000}
    )
    docs = db.similarity_search(question, k=10)
    prompt = """Answer the following QUESTION based on the CONTEXT
    given. If you do not know the answer and the CONTEXT doesn't
    contain the answer truthfully say "I don't know"

        CONTEXT:{context}
        QUESTION:{question}
        ANSWER:
        """

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt,
    )
    chain = LLMChain(llm=model, prompt=prompt_template)
    return prompt, docs, chain.run(context = docs, question=question)


def main():
    st.set_page_config(layout="wide")
    st.title("Custom Summarization App")
    user_prompt = st.text_input("Enter the user prompt")
    pdf_file_path = st.text_input("Enter the pdf file path")
    if pdf_file_path != "":
        db = pdf_to_faiss(pdf_file_path)
        st.write("Pdf was loaded successfully")
        if st.button("Summarize"):
            _,_,result = answer(db,user_prompt)
            st.write("Here is the answer to your question:")
            st.write(result)

if __name__ == "__main__":
    main()


