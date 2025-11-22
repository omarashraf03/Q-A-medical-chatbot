import streamlit as st
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.5-flash"
DOCS_PATH = "/Users/mac/Desktop/Q&A STAI/docs/"

llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.2, google_api_key=api_key)

@st.cache_resource
def setup_rag():
    loader = DirectoryLoader(DOCS_PATH, glob="**/*.*", show_progress=True)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    splits = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=api_key)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever()
retriever = setup_rag()

###########################################################################

SYSTEM_BASE = (
    "You are a safe medical assistant. Only answer questions on medicine, health, "
    "If CONTEXT is empty/irrelevant, use general medical knowledge. "
    "If non-medical, reply: '⚠️ I can only answer medical questions.'"
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{system_content}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}")
])

def get_context(input_data):
    return input_data["retriever"].invoke(input_data["question"])

def build_system_content(input_data):
    context_docs = input_data["context"]
    content_text = "\n\n".join(doc.page_content for doc in context_docs)
    return f"{SYSTEM_BASE}\n\nUse CONTEXT:\n{content_text}"

rag_chain = RunnablePassthrough.assign(
    context=get_context
).assign(
    system_content=build_system_content
) | PROMPT | llm

###############################################################

st.title("Q&A Medical Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

###############################################################

banned_words = ["suicide", "kill", "drugs", "violence"]
emergency_words = ["chest pain", "shortness of breath", "bleeding"]

def contains(words, text):
    text_lower = text.lower()
    for word in words:
        if word in text_lower:
            return True
    return False

###############################################################

def format_history(history_list):
    messages = []
    for chat in history_list:
        messages.append(HumanMessage(content=chat["user"]))
        messages.append(AIMessage(content=chat["ai"]))
    return messages

###############################################################

question = st.chat_input("Type your medical question here...")

if question:

    if contains(banned_words, question):
        st.warning("⚠️ Prohibited content.")
        st.stop()

    if contains(emergency_words, question):
        answer = "🚨 Emergency! Contact a doctor immediately."
        st.warning(answer)
        st.session_state.history.append({"user": question, "ai": answer})
        st.rerun()

    history_messages = format_history(st.session_state.history)

    with st.spinner("Analyzing..."):
        input_data = {
            "question": question,
            "retriever": retriever,
            "chat_history": history_messages
        }
        answer = rag_chain.invoke(input_data).content

    st.session_state.history.append({"user": question, "ai": answer})
    st.rerun()

for chat in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(chat["user"])
    with st.chat_message("assistant"):
        st.markdown(chat["ai"])
