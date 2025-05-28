# chatbot_core.py
import os
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import faiss
import numpy as np
import streamlit as st

import logging
logging.basicConfig(level=logging.INFO)


from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# ------ config ------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = groq_api_key
GROQ_MODEL = "llama3-70b-8192"

llm = ChatGroq(model_name=GROQ_MODEL, temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

system_template = (
    "You are a helpful AI assistant that answers questions based on web page content."
    "Respond in the first person, addressing the user directly as 'you'."
    "Keep your tone conversational and informative."
)

system_message = SystemMessagePromptTemplate.from_template(system_template)
human_template = """
User query: {query}

Web page content:
{content}

Answer the user's question based on the above content. Say "you" instead of "the user".
"""

human_message = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

# ------ web scraping and indexing ------
# def scrape_text_from_url(url):
#     try:
#         response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
#         soup = BeautifulSoup(response.content, 'html.parser')
#         for tag in soup(["script", "style", "noscript"]):
#             tag.decompose()
#         text = ' '.join(soup.stripped_strings)
#         logging.info(f"TEXT: {text}")
#         if any(bad in text.lower() for bad in [
#             "enable javascript", "cloudflare", "ad blocker", "access denied"
#         ]) or len(text) < 200:
#             return ""
#         return text
#     except:
#         return ""

def scrape_text_from_url(url):
    print("scrape_text_from_url")
    try:
        response = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0"
        })
        soup = BeautifulSoup(response.content, 'html.parser')

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        
        text = ' '.join(soup.stripped_strings)
        # logging.info(f"TEXT: {text}")
        text_lower = text.lower()

        block_phrases = [
            "enable javascript",
            "please enable javascript",
            "ad blocker",
            "disable your ad blocker",
            "cloudflare",
            "checking your browser before accessing",
            "are you human",
            "access denied"
        ]
        if any(phrase in text_lower for phrase in block_phrases) or len(text) < 200:
            logging.info(f"Blocked or unusable content from: {url}")
            return ""

        return text
    except Exception as e:
        logging.info(f"Error scraping {url}: {e}")
        return ""

def fetch_top_webpages(query: str, desired_count=5, max_attempts=15):
    print("fetch_top_webpages")
    try:
        print("try")
        results = list(search(query, 5)) #num=max_attempts, stop=max_attempts, pause=2.0))
        print("found")
        valid = []
        # logging.info(f"results: {results}")
        for url in results:
            content = scrape_text_from_url(url)
            if content:
                valid.append((url, content))
            if len(valid) >= desired_count:
                break
        print(valid)
        return valid
    except:
        return []

def chunk_text(text, chunk_size=500):
    print("chunk_text")
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_chunks(chunks, embedder):
    return embedder.encode(chunks, convert_to_tensor=False)

def create_faiss_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

def retrieve(query, chunks, index, embedder, k=3):
    print("hi")
    q_emb = embedder.encode([query])[0]
    D, I = index.search(np.array([q_emb]).astype('float32'), k)
    return I[0]

# ------ chat agent ------
# def chat_agent(user_input: str,) -> str:
def chat_agent(user_input: str, embedder) -> str:
    print("chat_agent")
    top_urls = fetch_top_webpages(user_input)
    if not top_urls:
        return "Sorry, no relevant search results found."

    all_chunks = []
    chunk_sources = []
    for url, content in top_urls:
        chunks = chunk_text(content)
        all_chunks.extend(chunks)
        chunk_sources.extend([url] * len(chunks))

    # logging.info("CONTENT:", all_chunks)

    # from sentence_transformers import SentenceTransformer
    # embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_chunks(all_chunks, embedder)
    index = create_faiss_index(embeddings)
    top_indices = retrieve(user_input, all_chunks, index, embedder)

    top_chunks = [all_chunks[i] for i in top_indices]
    top_sources = [chunk_sources[i] for i in top_indices]

    context = "\n".join(top_chunks)
    source_info = "\n".join(set(f"- {url}" for url in top_sources))

    # logging.info("CONTEXT:", context)

    inputs = {"query": user_input, "content": f"{context}\n\nSources:\n{source_info}"}
    history = memory.load_memory_variables({}).get("chat_history", [])
    messages = history + chat_prompt.format_prompt(**inputs).to_messages()
    response = llm(messages)

    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(response.content)
    return response.content.strip() + f"\n\nSources:\n" + "\n".join(set(top_sources))
