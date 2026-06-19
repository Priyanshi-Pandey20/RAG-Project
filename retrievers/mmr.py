from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


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

similarity_retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs ={"k":3}
)

print("Similarity Search")

similarity_docs = similarity_retriever.invoke("What is gradient descent?")

for doc in similarity_docs:
    print(doc.page_content)

mmr_retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs ={"k":3}
)  


print("MMR Results")
mmr_docs = mmr_retriever.invoke("What is gradient descent?")

for doc in mmr_docs:
    print(doc.page_content)

