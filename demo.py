from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import streamlit as st
import os, io, tempfile, warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize the model
llm = HuggingFaceHub(
    repo_id="google/flan-t5-xxl", 
    model_kwargs={"temperature": 1, "max_length": 100}, 
    task="text-generation"
)

# Function to process and retrieve the answer
def process_pdf(uploaded_pdf, query):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())  # Save uploaded content to temp file
        temp_file_path = tmp_file.name  # Get the temp file path

    # Load and split the PDF
    loader = PyPDFLoader(temp_file_path)
    pages = loader.load_and_split()

    # Define text splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=64,
        separators=[r'\n\n', r'\n', r'(?=>\. )', ' ', '']
    )
    docs = text_splitter.split_documents(pages)

    # Embeddings and vector store
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    # Create the QA chain
    # qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=db.as_retriever(search_kwargs={"k": 3})
    # )

    # Custom function to get the answer
    def answer_QA_chain(q):
        chain = load_qa_chain(llm, chain_type="stuff")
        docs = db.similarity_search(q, k=3)
        response = chain.run(input_documents=docs, question=q)
        start_index = response.find("Helpful Answer:")
        if start_index != -1:
            answer = response[start_index + len("Helpful Answer:"):].strip()
            return answer
        return response.strip()

    # Get the answer
    return answer_QA_chain(query)

# Streamlit UI
st.title("Quation Answer System To Own PDF")

# File upload
uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

# Question input
query = st.text_input("Enter your question:")

# Process the PDF and show the answer
if uploaded_pdf and query:
    with st.spinner("Processing your PDF and generating answer..."):
        answer = process_pdf(uploaded_pdf, query)
        st.write("### Answer:")
        st.write(answer)
