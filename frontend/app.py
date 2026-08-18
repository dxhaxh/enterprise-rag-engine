import streamlit as st
import requests

# 1. Page Configuration
st.set_page_config(page_title="Enterprise RAG", page_icon="🛡️", layout="centered")
st.title("🛡️ Enterprise Secure RAG")
st.markdown("Interact with the mathematically evaluated, RBAC-secured RAG pipeline.")

# 2. Sidebar Configuration (Simulating an enterprise environment)
with st.sidebar:
    st.header("⚙️ Security & Context")
    st.markdown("Modify these to test RBAC and Guardrails.")
    tenant_id = st.text_input("Tenant ID", value="tenant-Alpha")
    user_role = st.text_input("User Role", value="quant")
    top_k = st.slider("Context Chunks to Retrieve", min_value=1, max_value=5, value=2)

# 3. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Handle User Input
if prompt := st.chat_input("Ask the secure RAG engine a question..."):
    # Display user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 6. Call the FastAPI Backend
    with st.chat_message("assistant"):
        with st.spinner("Searching secure vector store..."):
            try:
                payload = {
                    "query": prompt,
                    "tenant_id": tenant_id,
                    "user_role": user_role,
                    "top_k": top_k
                }
                
                # Make the POST request to our local FastAPI server
                response = requests.post("http://127.0.0.1:8000/ask", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    if sources:
                        st.caption(f"**Sources Verified:** {', '.join(sources)}")
                    
                    # Save AI response to history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                else:
                    # If our Guardrails catch something, display the nice 400/500 error!
                    error_msg = f"**🚨 API Error {response.status_code}:** {response.json().get('detail', 'Unknown error')}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
            except requests.exceptions.ConnectionError:
                st.error("🚨 Cannot connect to backend. Is the FastAPI server running (`uvicorn src.api.main:app`)?")