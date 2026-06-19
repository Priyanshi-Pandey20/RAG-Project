from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()

docs = [
     Document(page_content="Gradient descent is an optimization algorithm."),
    Document(page_content="Gradient descent minimizes loss function."),
    Document(page_content="Gradient descent reduces error during training."),
    Document(page_content="Neural networks use gradient descent."),
    Document(page_content="Support Vector Machines are supervised learning algorithms."),
    Document(page_content="Cats are domestic animals."),    
]

embeddings = HuggingFaceEmbeddings()
vectorstore = Chroma.from_documents(docs,embeddings)

retriever = vectorstore.as_retriever()

llm = ChatMistralAI(model="mistral-small-latest")

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm = llm
)

query = "What is gradient descent?"

docs = multi_query_retriever.invoke(query)

print("Retrieved Document")
for doc in docs:
    print(doc.page_content)

