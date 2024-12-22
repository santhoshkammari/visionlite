import streamlit as st
from typing import List, Dict

from visionlite import vision

# Dummy data
dummy_data = [
    {"url": "https://example1.com", "top_k": "This is the first important piece of information about AI models"},
    {"url": "https://example2.com", "top_k": "Here's another key insight about machine learning applications"},
    {"url": "https://example3.com", "top_k": "The third point discusses neural networks and their applications"},
    {"url": "https://example4.com", "top_k": "Fourth item talks about computer vision developments"},
    {"url": "https://example5.com", "top_k": "Finally, this covers recent advances in NLP"}
]


def main():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

        .anthropic-container {
            max-width: 800px;
            margin: 0 auto;
            font-family: 'Inter', sans-serif;
            padding: 1.5rem 1rem;
        }

        .section-title {
            font-size: 14px;
            font-weight: 500;
            color: #666666;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 1rem;
        }

        .main-title {
            font-size: 28px;
            font-weight: 600;
            color: #111827;
            margin-bottom: 1.5rem;
            line-height: 1.3;
        }

        .result-item {
            padding: 0.5rem 0;
            border-bottom: 1px solid #EAECEF;
        }

        .result-text {
            color: #111827;
            font-size: 17px;
            line-height: 1.5;
            font-weight: 400;
        }

        .result-url {
            color: #FF7246;
            font-size: 14px;
            margin-top: 0.2rem;
            display: none;
            font-weight: 500;
        }

        .result-item:hover .result-url {
            display: block;
        }

        .result-item:hover .result-text {
            color: #FF7246;
        }

        .stApp {
            background-color: #FFFFFF;
        }

        .block-container {
            padding-top: 0;
            max-width: 1000px;
        }

        a {
            text-decoration: none !important;
        }

        .footer-text {
            color: #6B7280;
            font-size: 14px;
            margin-top: 1.5rem;
            padding-top: 0.75rem;
            border-top: 1px solid #EAECEF;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('''
        <div class="anthropic-container">
            <div class="section-title">Search Results</div>
            <h1 class="main-title">AI Research Insights</h1>
    ''', unsafe_allow_html=True)
    for item in vision(''):
        st.markdown(f"""
            <a href="{item['url']}" target="_blank">
                <div class="result-item">
                    <div class="result-text">{item['top_k']}</div>
                    <div class="result-url">Read more →</div>
                </div>
            </a>
        """, unsafe_allow_html=True)

    st.markdown('''
            <div class="footer-text">
                Results are sorted by relevance
            </div>
        </div>
    ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()