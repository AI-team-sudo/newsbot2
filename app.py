import streamlit as st
from openai import OpenAI
import pinecone
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI with secrets
client = OpenAI(api_key=st.secrets["openai_api_key"])

# Initialize Pinecone
pinecone_key = st.secrets["pinecone_api_key"]
pc = pinecone.Pinecone(api_key=pinecone_key)
index = pc.Index("newsbot2")

def truncate_text(text, max_length=300):
    """Truncate text to a maximum length while keeping whole words."""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + '...'

def enhance_query_with_gpt4(query, language="both"):
    system_prompt = """
    Enhance this news search query. Return in format:
    ORIGINAL_QUERY: <query>
    ENHANCED_QUERY: <improved query>
    SEARCH_TERMS: <3 most relevant terms>
    CONTEXT: <brief context>
    Be concise.
    """

    user_prompt = f"Query: {query}"

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )

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
            'search_terms': [term.strip() for term in result.get('SEARCH_TERMS', '').split(',')][:3],
            'context': result.get('CONTEXT', '')
        }
    except Exception as e:
        logger.error(f"Error in query enhancement: {str(e)}")
        st.error(f"Error in query enhancement: {str(e)}")
        return {
            'original_query': query,
            'enhanced_query': query,
            'search_terms': [],
            'context': ''
        }

def get_embedding(text):
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        st.error(f"Error generating embedding: {str(e)}")
        return None

def filter_results_with_gpt4(query_context, results, threshold=0.7):
    results_text = ""
    for i, match in enumerate(results.matches[:5]):
        title = truncate_text(match.metadata.get('title', ''), 100)
        content = truncate_text(match.metadata.get('content', ''), 200)
        results_text += f"\nArticle {i+1}:\nTitle: {title}\nContent: {content}\nScore: {match.score:.2f}\n"

    system_prompt = """
    Analyze these news articles for relevance to the query context.
    Return results in format:
    ARTICLE_START
    INDEX: <number>
    SCORE: <0-1>
    EXPLANATION: <brief explanation>
    ARTICLE_END
    Only include relevant articles above the threshold.
    """

    user_prompt = f"""
    Query Context: {truncate_text(query_context, 200)}
    Threshold: {threshold}
    Analyze these articles:
    {results_text}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )

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

    except Exception as e:
        logger.error(f"Error in filtering results: {str(e)}")
        st.error(f"Error in filtering results: {str(e)}")
        return {'results': [{'article_index': i+1, 'relevance_score': match.score,
                           'explanation': 'Direct match'}
                          for i, match in enumerate(results.matches[:5])]}

def semantic_search(query, top_k=5):
    enhanced_query_data = enhance_query_with_gpt4(query)
    search_terms = enhanced_query_data['search_terms'][:3]
    search_vectors = []

    main_embedding = get_embedding(enhanced_query_data['enhanced_query'])
    if main_embedding:
        search_vectors.append(main_embedding)

    for term in search_terms:
        if term:
            term_embedding = get_embedding(term)
            if term_embedding:
                search_vectors.append(term_embedding)

    if not search_vectors:
        st.error("Failed to generate embeddings for search")
        return None, None, None

    all_results = []
    for vector in search_vectors:
        try:
            results = index.query(
                vector=vector,
                top_k=min(top_k, 5),
                include_metadata=True
            )
            all_results.extend(results.matches)
        except Exception as e:
            logger.error(f"Error querying Pinecone: {str(e)}")
            st.error(f"Error querying Pinecone: {str(e)}")
            continue

    seen_ids = set()
    unique_results = []
    for result in all_results:
        if result.id not in seen_ids:
            seen_ids.add(result.id)
            unique_results.append(result)

    unique_results.sort(key=lambda x: x.score, reverse=True)
    unique_results = unique_results[:top_k]

    if not unique_results:
        st.warning("No results found")
        return enhanced_query_data, {'results': []}, []

    filtered_results = filter_results_with_gpt4(
        enhanced_query_data['context'],
        type('Results', (), {'matches': unique_results})()
    )

    return enhanced_query_data, filtered_results, unique_results

def log_metadata_issues(metadata, article_id):
    issues = []
    if not metadata.get('title'):
        issues.append('Missing title')
    if not metadata.get('content'):
        issues.append('Missing content')
    if not metadata.get('date'):
        issues.append('Missing date')
    if issues:
        logger.warning(f"Article {article_id} has metadata issues: {', '.join(issues)}")
        st.warning(f"Article {article_id} has metadata issues: {', '.join(issues)}")

# Streamlit UI
st.set_page_config(
    page_title="Enhanced Gujarati News Search",
    page_icon="📰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stTitle {
        font-size: 38px;
        font-weight: bold;
        color: #1E3D59;
        margin-bottom: 20px;
    }
    .news-card {
        padding: 20px;
        border-radius: 8px;
        background-color: #f8f9fa;
        margin: 15px 0;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .query-enhancement {
        background-color: #e9ecef;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .relevance-score {
        color: #28a745;
        font-weight: bold;
        margin: 10px 0;
    }
    .explanation {
        font-style: italic;
        color: #6c757d;
        padding: 10px 0;
    }
    .error-message {
        color: #dc3545;
        padding: 10px;
        border-radius: 5px;
        background-color: #f8d7da;
        margin: 10px 0;
    }
    .loading-spinner {
        text-align: center;
        padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("ગુજરાતી સમાચાર શોધ એન્જિન / Enhanced Gujarati News Search Engine")

# Search interface
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    query = st.text_input("Enter your search query (English or ગુજરાતી માં):", "")
with col2:
    top_k = st.number_input("Number of results:", min_value=1, max_value=10, value=5)
with col3:
    threshold = st.slider("Relevance threshold:", 0.0, 1.0, 0.7, 0.1)

# Search button
if st.button("Search", type="primary"):
    if not query:
        st.warning("Please enter a search query.")
    else:
        try:
            with st.spinner("🔍 Searching with enhanced AI capabilities..."):
                enhanced_query_data, filtered_results, all_results = semantic_search(query, top_k)

                if enhanced_query_data and filtered_results and all_results:
                    with st.expander("🔍 Search Enhancement Details"):
                        st.markdown("### Query Analysis")
                        st.markdown(f"**Original Query:** {enhanced_query_data['original_query']}")
                        st.markdown(f"**Enhanced Query:** {enhanced_query_data['enhanced_query']}")
                        st.markdown("**Related Search Terms:**")
                        for term in enhanced_query_data['search_terms']:
                            st.markdown(f"- {term}")
                        st.markdown(f"**Context:** {enhanced_query_data['context']}")

                    result_count = len(filtered_results.get('results', []))
                    st.subheader(f"Found {result_count} relevant results")

                    if result_count > 0:
                        for result in filtered_results['results']:
                            article_idx = result['article_index'] - 1
                            if article_idx < len(all_results):
                                article = all_results[article_idx]
                                metadata = article.metadata
                                log_metadata_issues(metadata, article.id)

                                # Clean and validate metadata
                                title = metadata.get('title', '').strip()
                                if not title:
                                    title = "Untitled Article"

                                date = metadata.get('date', '')
                                if date:
                                    try:
                                        parsed_date = datetime.strptime(date, '%Y-%m-%d')
                                        formatted_date = parsed_date.strftime('%B %d, %Y')
                                    except:
                                        formatted_date = date
                                else:
                                    formatted_date = 'Date not available'

                                content = metadata.get('content', '').strip()
                                if not content:
                                    content = "Content not available"

                                link = metadata.get('link', '#')
                                link_text = f"""<p><a href="{link}" target="_blank">
                                    વધુ વાંચો (Read more) →
                                </a></p>""" if link and link != '#' else ""

                                st.markdown(f"""
                                    <div class="news-card">
                                        <h3>{title}</h3>
                                        <p><em>{formatted_date}</em></p>
                                        <p>{content}</p>
                                        {link_text}
                                        <div class="relevance-score">
                                            Relevance Score: {result['relevance_score']:.2f}
                                        </div>
                                        <div class="explanation">
                                            {result['explanation']}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.info("No results met the relevance threshold. Try adjusting the threshold or modifying your search query.")

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            st.error(f"""
                An error occurred while processing your search.
                Error details: {str(e)}
                Please try:
                1. Refining your search query
                2. Reducing the number of requested results
                3. Trying again in a few moments
            """)

# Sidebar
with st.sidebar:
    st.markdown("### Search Tips")
    st.markdown("""
    - Use specific keywords for better results
    - Try both English and Gujarati queries
    - Adjust the relevance threshold for more/fewer results
    - Check the query enhancement details for better understanding
    """)

    st.markdown("### About")
    st.markdown("""
    This enhanced search engine uses:
    - Cosine Similarity and LLM for query understanding
    - Advanced semantic search
    - Multi-vector matching
    - Relevance filtering
    - Bilingual support
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>Powered by Embeddings and Pinecone | Last updated: February 2024</small>
</div>
""", unsafe_allow_html=True)

# Created/Modified files during execution:
print("Created/Modified files: streamlit_app.py")
