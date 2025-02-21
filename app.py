import streamlit as st
from openai import OpenAI
import pinecone
from datetime import datetime

# Initialize OpenAI with secrets
client = OpenAI(api_key=st.secrets["openai_api_key"])

# Initialize Pinecone with the latest method
pinecone_key = st.secrets["pinecone_api_key"]
pc = pinecone.Pinecone(api_key=pinecone_key)

# Connect to your index
index = pc.Index("newsbot2")

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def search_news(query, top_k=5):
    # Get embedding for the query
    query_embedding = get_embedding(query)

    # Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    return results

def format_date(date_str):
    try:
        date_obj = datetime.strptime(date_str, "%d-%m-%Y | %I:%M %p")
        return date_obj.strftime("%d %B %Y, %I:%M %p")
    except:
        return date_str

# Streamlit UI
st.set_page_config(
    page_title="Gujarati News Search",
    page_icon="📰",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stTitle {
        font-size: 42px;
        font-weight: bold;
        color: #1E3D59;
    }
    .news-card {
        padding: 20px;
        border-radius: 5px;
        background-color: #f8f9fa;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("ગુજરાતી સમાચાર શોધ એન્જિન / Gujarati News Search Engine")

# Search interface
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Enter your search query (English or ગુજરાતી માં):", "")
with col2:
    top_k = st.number_input("Number of results:", min_value=1, max_value=10, value=5)

if st.button("Search", type="primary"):
    if query:
        try:
            with st.spinner("Searching..."):
                results = search_news(query, top_k)

                if results.matches:
                    for i, match in enumerate(results.matches, 1):
                        metadata = match.metadata
                        score = match.score

                        # Create a card-like container for each result
                        st.markdown(f"""
                            <div class="news-card">
                                <h3>{metadata['title']}</h3>
                                <p><em>{format_date(metadata['date'])}</em></p>
                                <p>{metadata['content']}</p>
                                <p><a href="{metadata['link']}" target="_blank">વધુ વાંચો (Read more)</a></p>
                                <p><small>Relevance Score: {score:.2f}</small></p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("No results found.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a search query.")

# Footer
st.markdown("---")
with st.expander("About"):
    st.write("""
    This news search engine allows you to search through Gujarati news articles using either English or Gujarati queries.
    The search is powered by OpenAI's embeddings and Pinecone's vector database.
    """)

# Created/Modified files during execution:
print("Created/Modified files: streamlit_app.py")
