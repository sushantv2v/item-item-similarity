import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import streamlit as st


@st.cache_data(ttl=600)  # Set the time-to-live (TTL) in seconds
def load_data():
    base_loc = ''
    # Load the ratings data
    u_data = pd.read_csv(base_loc + 'Data/u.data', sep='\t', header=None,
                         names=['user_id', 'item_id', 'rating', 'timestamp'])

    # Load the movie data
    u_item = pd.read_csv(base_loc + 'Data/u.item', sep='|', encoding='latin-1', header=None,
                         names=['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL',
                                'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                                'Documentary',
                                'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                                'Sci-Fi', 'Thriller', 'War', 'Western'])

    # Pivot the data to get a user-item matrix
    user_item_matrix = u_data.pivot(index='item_id', columns='user_id', values='rating').fillna(0)

    # Convert to sparse matrix
    user_item_sparse_matrix = csr_matrix(user_item_matrix.values)

    # Calculate the cosine similarity between items
    item_item_similarity = cosine_similarity(user_item_sparse_matrix)

    return u_item, user_item_matrix, item_item_similarity


def get_similar_items(item_id, k=10, min_ratings=5, item_item_similarity=None):
    if item_item_similarity is None:
        u_item, _, item_item_similarity = load_data()

    num_ratings = np.sum(user_item_matrix > 0, axis=1)

    if num_ratings[item_id] >= min_ratings:
        # Use item-item similarity
        similarity = item_item_similarity
    else:
        # Use metadata similarity as a fallback
        similarity = item_metadata_similarity

    # Get the top K similar items
    similar_item_ids = np.argsort(similarity[item_id])[::-1][:k]

    # Remove the item itself from the list
    similar_item_ids = similar_item_ids[similar_item_ids != item_id]

    # Return the similar item IDs and their similarity scores
    return similar_item_ids


def get_movie_title(item_id, u_item=None):
    if u_item is None:
        u_item, _ = load_data()
    return u_item[u_item['item_id'] == item_id]['movie_title'].values[0]


# Streamlit app
st.title('Movie Recommendation System')

# Load the data and cache it
u_item, user_item_matrix, item_item_similarity = load_data()

# Movie selection dropdown with search functionality
selected_movie_title = st.selectbox('Search and select a movie:', u_item['movie_title'])

# Get selected movie_id
selected_movie_id = u_item[u_item['movie_title'] == selected_movie_title]['item_id'].values[0]

# Get similar movies
similar_item_ids = get_similar_items(selected_movie_id, item_item_similarity=item_item_similarity)
similar_item_titles = [get_movie_title(item_id, u_item=u_item) for item_id in similar_item_ids]

# Display similar movies
st.write(f"Similar movies to '{selected_movie_title}':")
for title in similar_item_titles:
    st.write(f"- {title}")
