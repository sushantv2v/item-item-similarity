import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import streamlit as st

# Load the ratings data
u_data = pd.read_csv('Data/u.data', sep='\t', header=None,
                     names=['user_id', 'item_id', 'rating', 'timestamp'])

# Load the movie data
u_item = pd.read_csv('Data/u.item', sep='|', encoding='latin-1', header=None,
                     names=['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL',
                            'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
                            'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                            'Sci-Fi', 'Thriller', 'War', 'Western'])

# Pivot the data to get a user-item matrix
user_item_matrix = u_data.pivot(index='item_id', columns='user_id', values='rating').fillna(0)

# Convert to sparse matrix
user_item_sparse_matrix = csr_matrix(user_item_matrix.values)

# Calculate the cosine similarity between items
item_item_similarity = cosine_similarity(user_item_sparse_matrix)

# Now, let's create a similarity matrix based on genre
item_metadata = u_item.drop(columns=['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'])
item_metadata_sparse_matrix = csr_matrix(item_metadata.values)
item_metadata_similarity = cosine_similarity(item_metadata_sparse_matrix)


def get_similar_items(item_id, k=10, min_ratings=5):
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
    return similar_item_ids, similarity[item_id, similar_item_ids]


def get_movie_title(item_id):
    return u_item[u_item['item_id'] == item_id]['movie_title'].values[0]


# Streamlit app
st.title('Movie Recommendation System')

# Movie selection dropdown with search functionality
selected_movie_title = st.selectbox('Search and select a movie:', u_item['movie_title'])

try:
    # Get selected movie_id if it exists
    selected_movie_ids = u_item[u_item['movie_title'] == selected_movie_title]['item_id'].values
    if len(selected_movie_ids) > 0:
        selected_movie_id = selected_movie_ids[0]
        # Get similar movies
        similar_item_ids, _ = get_similar_items(selected_movie_id)
        # Get the movie titles for the similar item_ids
        similar_item_movie_titles = [get_movie_title(item_id) for item_id in similar_item_ids]
        # Display similar movies
        st.write(f"Similar movies to '{selected_movie_title}':")
        for title in similar_item_movie_titles:
            st.write(f"- {title}")
    else:
        st.write(f"No similar movies found for '{selected_movie_title}'.")
except IndexError:
    st.write(f"Error: '{selected_movie_title}' is not found in the movie list.")
