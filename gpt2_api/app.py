import streamlit as st
import requests

st.set_page_config(page_title="Text Generator", layout="centered")

st.title("🧠 Text Generation with GPT-like Transformer Model")

# User input
prompt = st.text_area("Enter your prompt", height=150)

max_length = st.slider("Max Length", min_value=10, max_value=500, value=100, step=10)

if st.button("Generate"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating response..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/generate",
                    json={"prompt": prompt, "max_length": max_length}
                )
                if response.status_code == 200:
                    output = response.json()["response"]
                    st.subheader("Generated Text")
                    st.write(output)
                else:
                    st.error(f"Error: {response.json()['detail']}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
