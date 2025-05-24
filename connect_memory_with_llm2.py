import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq  # NEW import

# Step 1: Setup LLM (Mistral-7B via Groq)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def load_llm():
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="gemma2-9b-it",  # You can also try "llama3-70b-8192" or "gemma-7b-it"
        temperature=0.5,
        max_tokens=512
    )
    return llm

# Step 2: Create Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don’t know the answer, just say you don’t know—don’t try to make it up.
Do not provide anything outside of the given context.

Context:
{context}

Question:
{question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

# Step 3: Load FAISS Vectorstore
DB_FAISS_PATH = 'vectorstore/db_faiss'
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Create RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 5: Ask a question
user_query = input("Write Query Here: ")
response = qa_chain.invoke({"query": user_query})

# Step 6: Print results
print("\nRESULT:\n", response["result"])
print("\nSOURCE DOCUMENTS:\n", response["source_documents"])
