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
    Consider:
    1. Related terms and synonyms
    2. Both English and Gujarati contexts
    3. Current events context
    4. Return a JSON with both original and enhanced queries
    """

    user_prompt = f"""
    Original query: {query}
    Language: {language}

    Generate enhanced search terms that will help find relevant news articles.
    Return in JSON format with keys:
    - original_query
    - enhanced_query
    - search_terms (list of related terms)
    - context (brief context about the query)
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={ "type": "json_object" }
    )

    return json.loads(response.choices[0].message.content)

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def filter_results_with_gpt4(query_context, results, threshold=0.7):
    # Prepare results for GPT-4 analysis
    results_text = ""
    for i, match in enumerate(results.matches):
        results_text += f"\nArticle {i+1}:\nTitle: {match.metadata['title']}\nContent: {match.metadata['content']}\nScore: {match.score}\n"

    system_prompt = """
    You are a news relevance expert. Analyze the search results and determine their relevance to the original query context.
    Return a JSON array of relevant article indices with relevance scores and explanations.
    """

    user_prompt = f"""
    Query Context: {query_context}

    Search Results:
    {results_text}

    Analyze each article's relevance to the query context.
    Return JSON with:
    - article_index (1-based)
    - relevance_score (0-1)
    - explanation
    Only include articles with relevance score > {threshold}
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={ "type": "json_object" }
    )

    return json.loads(response.choices[0].message.content)

def semantic_search(query, top_k=5):
    # Enhance query using GPT-4
    enhanced_query_data = enhance_query_with_gpt4(query)

    # Get embeddings for enhanced query and search terms
    search_vectors = []
    search_vectors.append(get_embedding(enhanced_query_data['enhanced_query']))
    for term in enhanced_query_data['search_terms']:
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
                for result in filtered_results['results']:
                    article_idx = result['article_index'] - 1
                    if article_idx < len(all_results):
                        article = all_results[article_idx]
                        metadata = article.metadata

                        st.markdown(f"""
                            <div class="news-card">
                                <h3>{metadata['title']}</h3>
                                <p><em>{metadata['date']}</em></p>
                                <p>{metadata['content']}</p>
                                <p><a href="{metadata['link']}" target="_blank">વધુ વાંચો (Read more)</a></p>
                                <p><small>Relevance Score: {result['relevance_score']:.2f}</small></p>
                                <p><small>Explanation: {result['explanation']}</small></p>
                            </div>
                            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a search query.")

# Footer
st.markdown("---")
with st.expander("About"):
    st.write("""
    This enhanced news search engine uses GPT-4 for:
    1. Query understanding and enhancement
    2. Semantic relevance filtering
    3. Multi-vector search with related terms
    4. Detailed relevance explanations

    The search is powered by OpenAI's GPT-4, embeddings, and Pinecone's vector database.
    """)

# Created/Modified files during execution:
print("Created/Modified files: streamlit_app.py")
