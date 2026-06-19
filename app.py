# app.py

from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
import tempfile

load_dotenv()

st.set_page_config(
    page_title="RAG Book Assistant",
    page_icon="📚"
)

st.title("📚 RAG Book Assistant")
st.write("Upload a PDF book and ask questions about it.")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload a Book (PDF)",
    type=["pdf"]
)

# Create Database
if uploaded_file is not None:

    if st.button("Process Book"):

        with st.spinner("Processing Book..."):

            # Save uploaded PDF temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(uploaded_file.read())
                pdf_path = temp_file.name

            # Load PDF
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            # Create embeddings
            embedding_model = MistralAIEmbeddings()

            # Store in Chroma
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory="chroma_db"
            )

            st.success("Book uploaded successfully!")

# Load existing database
try:

    embedding_model = MistralAIEmbeddings()

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )

    st.sidebar.success(
        f"Chunks Stored: {vectorstore._collection.count()}"
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    query = st.text_input(
        "Ask a Question"
    )

    if query:

        docs = retriever.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful AI assistant.
                    Use ONLY the provided context to answer the question.
                    If the answer is not present in the context,
                    say 'I could not find the answer in the document.'
                    """
                ),
                (
                    "human",
                    """Context:
                    {context}

                    Question:
                    {question}
                    """
                )
            ]
        )

        llm = ChatMistralAI(
            model="mistral-small-2506"
        )

        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": query
            }
        )

        response = llm.invoke(final_prompt)

        st.subheader("Answer")
        st.write(response.content)

except:
    pass