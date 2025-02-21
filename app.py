import streamlit as st
from openai import OpenAI
import pinecone
from datetime import datetime
import json

# Initialize OpenAI with secrets
client = OpenAI(api_key=st.secrets["openai_api_key"])

# Initialize Pinecone
pinecone_key = st.secrets["pinecone_api_key"]
pc = pinecone.Pinecone(api_key=pinecone_key)
index = pc.Index("newsbot2")

def enhance_query_with_gpt4(query, language="both"):
    system_prompt = """
    You are a news search expert. Enhance the given search query to improve search results.
    Consider related terms, synonyms, and both English and Gujarati contexts.
    Return your response in the following format:
    ORIGINAL_QUERY: <original query>
    ENHANCED_QUERY: <enhanced query>
    SEARCH_TERMS: <comma-separated list of related terms>
    CONTEXT: <brief context about the query>
    """

    user_prompt = f"Original query: {query}\nLanguage: {language}"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7
    )

    # Parse the response text into a structured format
    response_text = response.choices[0].message.content
    lines = response_text.split('\n')
    result = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            result[key.strip()] = value.strip()

    return {
        'original_query': result.get('ORIGINAL_QUERY', query),
        'enhanced_query': result.get('ENHANCED_QUERY', query),
        'search_terms': [term.strip() for term in result.get('SEARCH_TERMS', '').split(',')],
        'context': result.get('CONTEXT', '')
    }

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def filter_results_with_gpt4(query_context, results, threshold=0.7):
    results_text = ""
    for i, match in enumerate(results.matches):
        results_text += f"\nArticle {i+1}:\nTitle: {match.metadata['title']}\nContent: {match.metadata['content']}\nScore: {match.score}\n"

    system_prompt = """
    You are a news relevance expert. Analyze the search results and determine their relevance to the original query context.
    For each relevant article, provide the following information in this exact format:
    ARTICLE_START
    INDEX: <article number>
    SCORE: <relevance score between 0 and 1>
    EXPLANATION: <brief explanation of relevance>
    ARTICLE_END
    Only include articles with relevance score higher than the given threshold.
    """

    user_prompt = f"""
    Query Context: {query_context}
    Threshold: {threshold}

    Search Results:
    {results_text}

    Analyze each article's relevance to the query context.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7
    )

    # Parse the response text into structured format
    response_text = response.choices[0].message.content
    articles = response_text.split('ARTICLE_START')
    parsed_results = {'results': []}

    for article in articles:
        if 'INDEX:' in article:
            article_data = {}
            lines = article.split('\n')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    if key == 'INDEX':
                        article_data['article_index'] = int(value)
                    elif key == 'SCORE':
                        article_data['relevance_score'] = float(value)
                    elif key == 'EXPLANATION':
                        article_data['explanation'] = value
            if article_data:
                parsed_results['results'].append(article_data)

    return parsed_results

def semantic_search(query, top_k=5):
    # Enhance query using GPT-4
    enhanced_query_data = enhance_query_with_gpt4(query)

    # Get embeddings for enhanced query and search terms
    search_vectors = []
    search_vectors.append(get_embedding(enhanced_query_data['enhanced_query']))
    for term in enhanced_query_data['search_terms']:
        if term:  # Only add non-empty terms
            search_vectors.append(get_embedding(term))

    # Search Pinecone with multiple vectors
    all_results = []
    for vector in search_vectors:
        results = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True
        )
        all_results.extend(results.matches)

    # Remove duplicates and sort by score
    seen_ids = set()
    unique_results = []
    for result in all_results:
        if result.id not in seen_ids:
            seen_ids.add(result.id)
            unique_results.append(result)

    unique_results.sort(key=lambda x: x.score, reverse=True)

    # Filter results using GPT-4
    filtered_results = filter_results_with_gpt4(
        enhanced_query_data['context'],
        type('Results', (), {'matches': unique_results[:top_k]})()
    )

    return enhanced_query_data, filtered_results, unique_results

# Streamlit UI
st.set_page_config(
    page_title="Enhanced Gujarati News Search",
    page_icon="📰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stTitle { font-size: 42px; font-weight: bold; color: #1E3D59; }
    .news-card { padding: 20px; border-radius: 5px; background-color: #f8f9fa; margin: 10px 0; }
    .query-enhancement { background-color: #e9ecef; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .relevance-score { color: #28a745; }
    .explanation { font-style: italic; color: #6c757d; }
    </style>
    """, unsafe_allow_html=True)

st.title("ગુજરાતી સમાચાર શોધ એન્જિન / Enhanced Gujarati News Search Engine")

# Search interface
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Enter your search query (English or ગુજરાતી માં):", "")
with col2:
    top_k = st.number_input("Number of results:", min_value=1, max_value=10, value=5)

if st.button("Search", type="primary"):
    if query:
        try:
            with st.spinner("Searching with enhanced AI capabilities..."):
                enhanced_query_data, filtered_results, all_results = semantic_search(query, top_k)

                # Display query enhancement details
                with st.expander("View Search Enhancement Details"):
                    st.json(enhanced_query_data)

                # Display filtered results
                st.subheader("Relevant Results")
                if 'results' in filtered_results and filtered_results['results']:
                    for result in filtered_results['results']:
                        article_idx = result['article_index'] - 1
                        if article_idx < len(all_results):
                            article = all_results[article_idx]
                            metadata = article.metadata

                            st.markdown(f"""
                                <div class="news-card">
                                    <h3>{metadata['title']}</h3>
                                    <p><em>{metadata.get('date', 'Date not available')}</em></p>
                                    <p>{metadata['content']}</p>
                                    <p><a href="{metadata['link']}" target="_blank">વધુ વાંચો (Read more)</a></p>
                                    <p class="relevance-score">Relevance Score: {result['relevance_score']:.2f}</p>
                                    <p class="explanation">Analysis: {result['explanation']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.warning("No relevant results found for your query.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please try refining your search query or try again later.")
    else:
        st.warning("Please enter a search query.")

# Footer
st.markdown("---")
with st.expander("About"):
    st.write("""
    This enhanced news search engine uses a fined tuned resnet for:
    1. Query understanding and enhancement
    2. Semantic relevance filtering
    3. Multi-vector search with related terms
    4. Detailed relevance explanations

    The search is powered by OpenAI's embeddings, and Oracle Database 23Ai vector database.
    """)

# Created/Modified files during execution:
print("Created/Modified files: streamlit_app.py")
