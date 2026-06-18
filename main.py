from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()



template = ChatPromptTemplate.from_messages(
    [("system","you are a AI that summarizes a text"),
     ("human","{data}")
     ]
)

model =ChatMistralAI(model = "mistral-small-2506")


