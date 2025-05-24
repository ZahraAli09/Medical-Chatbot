import streamlit as st
import os
import json
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq


os.environ["CUDA_VISIBLE_DEVICES"] = ""

st.set_page_config( page_icon="🩺", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #2E86AB;'>🩺 HealthHelp AI: Ask, Learn, Stay Informed</h1>
    <p style='text-align: center; font-size:18px;'>Ask medical questions and get accurate, evidence-based answers instantly.</p>
    <hr style='border: 1px solid #2E86AB;'/>
    """, 
    unsafe_allow_html=True
)

# Sidebar buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

with col2:
    if st.button("📋 View History"):
        st.session_state.show_history = not st.session_state.get('show_history', False)

# Initialize session state for prompt history
if 'prompt_history' not in st.session_state:
    st.session_state.prompt_history = []

if 'show_history' not in st.session_state:
    st.session_state.show_history = False

# Display prompt history if toggled
if st.session_state.show_history:
    st.sidebar.markdown("### 📋 Prompt History")
    if st.session_state.prompt_history:
        # Add clear history button
        if st.sidebar.button("🗑️ Clear History"):
            st.session_state.prompt_history = []
            st.rerun()
        
        # Display history in sidebar
        for i, entry in enumerate(reversed(st.session_state.prompt_history[-10:])):  # Show last 10 entries
            with st.sidebar.expander(f"Query {len(st.session_state.prompt_history) - i}: {entry['timestamp']}"):
                st.write("**Question:**")
                st.write(entry['prompt'])
                if entry.get('response'):
                    st.write("**Answer:**")
                    st.write(entry['response'])
                else:
                    st.write("*Response not available*")
    else:
        st.sidebar.write("No prompts in history yet.")

st.markdown("""
<hr>
<div style='text-align: center; color: gray; font-size: 14px;'>
    ⚠️ <strong>Disclaimer:</strong> This chatbot is an AI assistant trained on medical literature. 
    It is intended for informational and educational purposes only. Do not rely solely on it for medical advice, diagnosis, or treatment. 
    Always consult a qualified healthcare professional for medical concerns.
</div>
""", unsafe_allow_html=True)

# Function to save prompt to history
def save_prompt_to_history(prompt):
    """Save user prompt to session state history with timestamp"""
    history_entry = {
        'prompt': prompt,
        'response': None,  # Will be updated later
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.prompt_history.append(history_entry)
    return len(st.session_state.prompt_history) - 1  # Return index for updating response later

def update_history_with_response(index, response):
    """Update the history entry with the AI response"""
    if 0 <= index < len(st.session_state.prompt_history):
        st.session_state.prompt_history[index]['response'] = response

# Function to export history (optional feature)
def export_history_to_json():
    """Export prompt history to JSON format"""
    if st.session_state.prompt_history:
        return json.dumps(st.session_state.prompt_history, indent=2)
    return "No history to export"

# Add export functionality to sidebar
if st.session_state.show_history and st.session_state.prompt_history:
    if st.sidebar.button("📥 Export History"):
        history_json = export_history_to_json()
        st.sidebar.download_button(
            label="💾 Download History JSON",
            data=history_json,
            file_name=f"medical_chatbot_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt


def load_llm(GROQ_API_KEY, model_name="gemma2-9b-it"):
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=model_name,  # You can also try "llama3-70b-8192" or "gemma-7b-it"
        temperature=0.5,
        max_tokens=512
    )
    return llm

def main():
    #st.title('Ask Your Virtual Medical Expert')

    if 'messages' not in st.session_state:
        st.session_state.messages=[]

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input('Pass your Prompt here:')

    if prompt:
        # Save prompt to history before processing
        history_index = save_prompt_to_history(prompt)
        
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content':prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you don't know the answer, just say you don't know—don't try to make it up.
                Do not provide anything outside of the given context.

                Context:
                {context}

                Question:
                {question}

                Start the answer directly. No small talk please.
                """
        model_name="gemma2-9b-it"
        GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
        
        try:
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error('Failed to load the vectorstore')

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(GROQ_API_KEY=GROQ_API_KEY, model_name=model_name),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)})


                
            response=qa_chain.invoke({'query':prompt})

            result=response["result"]
            #source_documents=response["source_documents"]
            result_to_show=result

            #result_to_show=result+"\nSource Docs:\n"+str(source_documents)
            #response="Hi, I am MediBot!"
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})
            
            # Update history with the response after displaying it
            update_history_with_response(history_index, result_to_show)

        except Exception as e:
            st.error(f"Error: {str(e)}")
            # Update history with error message
            update_history_with_response(history_index, f"Error occurred: {str(e)}")

    # Display prompt history statistics in the main area (optional)
    if st.session_state.prompt_history:
        st.sidebar.markdown(f"**Total Queries Asked:** {len(st.session_state.prompt_history)}")


if __name__ == '__main__':
    main()