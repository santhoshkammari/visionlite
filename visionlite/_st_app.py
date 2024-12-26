import streamlit as st
import time
from visionlite import minivisionai, deepvisionai, visionai, wordllama_qa
import pyperclip


def simulate_streaming(text):
    """Simulate streaming output"""
    chunks = text.split('\n')
    for chunk in chunks:
        if chunk.strip():
            yield chunk + '\n'
            time.sleep(0.04)


def copy_to_clipboard(text):
    """Copy text to clipboard"""
    pyperclip.copy(text)
    st.success("Copied to clipboard!")


def app():
    st.set_page_config(page_title="Vision AI Search", page_icon="🔍", layout="wide")

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for chat history
    with st.sidebar:
        st.header("💬 Chat History")
        if st.session_state.chat_history:
            for i, (query, result) in enumerate(st.session_state.chat_history):
                with st.expander(f"🔍 {query[:50]}..." if len(query) > 50 else f"🔍 {query}"):
                    st.write(result)
        else:
            st.info("No chat history yet")

    st.title("Vision AI Search Interface")
    # Store the clicked query
    if 'clicked_query' not in st.session_state:
        st.session_state.clicked_query = None

    query = st.text_area(
        "Enter your search query:",
        value=st.session_state.clicked_query if st.session_state.clicked_query else "",
        height=100
    )

    # Configuration buttons row
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        with st.popover("Search Type"):
            model_type = st.radio(
                "Select Search Type",
                ["Mini Vision", "Standard Vision", "Deep Vision","WordLLama Retreiver"],
                label_visibility="collapsed"
            )

    with col2:
        with st.popover("Model Settings"):
            model_names = [
                "llama3.2:1b-instruct-q4_K_M",
                "llama3.2:1b-instruct-fp16",
                "llama3.2:3b-instruct-q2_K",
                "llama3.2:3b-instruct-fp16",
                "llama3.2:latest",
                "llama3.1:8b",
                "llama3.1:8b-instruct-q4_0",
                "llama3.1:latest",
                "qwen2.5:0.5b-instruct",
                "qwen2.5:1.5b-instruct",
                "qwen2.5:3b-instruct",
                "qwen2.5:7b-instruct"
            ]

            # Model family selector
            model_name = st.radio(
                "Model Names",
                options=model_names,
                horizontal=True
            )
            model = st.text_input("Model Name", value=model_name)
            base_url = st.text_input("Base URL", value="http://localhost:11434")
            temperature = st.slider("Temperature",
                                    value=0.1 if model_type != "Deep Vision" else 0.05,
                                    min_value=0.0, max_value=1.0, step=0.05)

    with col3:
        with st.popover("Advanced Parameters"):
            col_left, col_right = st.columns(2)

            with col_left:
                max_urls = st.number_input("Max URLs",
                                           value=5 if model_type == "Mini Vision" else 10 if model_type == "Standard Vision" else 15,
                                           min_value=1, max_value=50)
                k = st.number_input("Top K Results",
                                    value=5 if model_type == "Mini Vision" else 5 if model_type == "Standard Vision" else 10,
                                    min_value=1, max_value=20)
                max_retries = st.number_input("Max Retries",
                                              value=3 if model_type == "Mini Vision" else 5 if model_type == "Standard Vision" else 10,
                                              min_value=1, max_value=20)
                animation = st.toggle("Enable Animation", value=False)
                allow_pdf = st.toggle("Allow PDF Extraction", value=True)

            with col_right:
                genai_query_k = st.number_input("GenAI Query K",
                                                value=3 if model_type == "Mini Vision" else 5 if model_type == "Standard Vision" else 7,
                                                min_value=1, max_value=20)
                query_k = st.number_input("Query K",
                                          value=5 if model_type == "Mini Vision" else 5 if model_type == "Standard Vision" else 15,
                                          min_value=1, max_value=20)
                allow_youtube = st.toggle("Allow YouTube", value=False)
                return_type = st.radio("Return Type", ["str", "list"])

    # Clear the clicked query after it's been used
    if st.session_state.clicked_query and query != st.session_state.clicked_query:
        st.session_state.clicked_query = None


    if st.button("Search", type="primary", use_container_width=True):
        if not query:
            st.warning("Please enter a search query.")
            return

        output_placeholder = st.empty()

        with st.spinner("Searching..."):
            if model_type == "Mini Vision":
                vision_func = minivisionai
            elif model_type == "Deep Vision":
                vision_func = deepvisionai
            else:
                vision_func = visionai

            try:
                if model_type == "WordLLama Retreiver":
                    result = wordllama_qa(query,k=k)
                else:
                    result = vision_func(
                        query=query,
                        max_urls=max_urls,
                        k=k,
                        model=model,
                        base_url=base_url,
                        temperature=temperature,
                        max_retries=max_retries,
                        animation=animation,
                        allow_pdf_extraction=allow_pdf,
                        allow_youtube_urls_extraction=allow_youtube,
                        genai_query_k=genai_query_k,
                        query_k=query_k,
                        return_type=return_type
                    )

                # Stream the results
                full_response = ""
                for chunk in simulate_streaming(result):
                    full_response += chunk
                    output_placeholder.markdown(full_response)

                # Add copy button after results
                st.button("Copy to Clipboard",
                          on_click=copy_to_clipboard,
                          args=(full_response,))

                # Add to chat history
                st.session_state.chat_history.append((query, full_response))

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")