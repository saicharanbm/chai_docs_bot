from dotenv import load_dotenv
load_dotenv()
import streamlit as st 
from langchain_qdrant import QdrantVectorStore
from configuration import COLLECTION_NAME
from helper import initialize_clients_for_indexing as initialize_clients, get_chat_response

@st.cache_resource
def load_clients():
    return initialize_clients()


# initialize an embedding model instance
embedding_model, text_splitter, qdrant_host, qdrant_api_key,console  = load_clients()
# --- Main Interface ---
st.title("🤖 Chai Docs Bot")


# --- Custom CSS for footer positioning ---
st.markdown("""
<style>
/* Add padding to the main container to make room for footer */
.main > div {
    padding-bottom: 120px !important;
}

/* Style the chat input container */
.stChatInput {
    margin-bottom: 30px !important;
    
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: rgba(38, 39, 48, 0.95);
    color: #fafafa;
    text-align: center;
    padding: 15px 0;
    z-index: 999;
    border-top: 1px solid #333;
    backdrop-filter: blur(10px);
}
.footer a {
    color: #58a6ff;
    text-decoration: none;
}
.footer a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# Footer with fixed positioning
st.markdown(
    '<div class="footer">Built with ❤️ by <a href="https://github.com/saicharanbm/chai_docs_bot" target="_blank">Sai Charan B M</a></div>',
    unsafe_allow_html=True
)


# --- Chat Interface Function ---
def chat_interface():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    vector_store = QdrantVectorStore.from_existing_collection(
                        url=qdrant_host,
                        api_key=qdrant_api_key,
                        collection_name=COLLECTION_NAME,
                        embedding=embedding_model
                    )

                    response = get_chat_response(prompt, vector_store)
                    st.markdown(response)

                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


# Mode selection
# st.header("Chat with Chai Code Docs :")
chat_interface()