import streamlit as st
from rag_pipeline import answer_from_url

st.set_page_config(
    page_title="Gemini 2.5 Flash RAG App",
    layout="centered"
)

st.title(" URL-Based RAG GenAI App")
st.caption("Powered by google.genai · Gemini 2.5 Flash")

url = st.text_input("Enter source URL")
question = st.text_area("Enter your question", height=120)
debug_mode = st.checkbox("Show debug output", value=True)

if st.button("Generate Answer"):
    if not url or not question:
        st.warning("Please provide both URL and question.")
    else:
        with st.spinner("Retrieving content and generating answer..."):
            if debug_mode:
                st.info("Debug mode: Check console/terminal for detailed logs")
            answer = answer_from_url(url, question, debug=debug_mode)

        st.subheader(" Answer (Grounded in URL)")
        st.write(answer)
