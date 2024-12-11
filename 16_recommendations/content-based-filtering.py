import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie data
movies_data = {
    'movie_id': [1, 2, 3, 4, 5],
    'title': [
        'The Dark Knight',
        'Inception',
        'Interstellar',
        'The Matrix',
        'Avatar'
    ],
    'genres': [
        'Action, Crime, Drama',
        'Action, Sci-Fi, Thriller',
        'Adventure, Drama, Sci-Fi',
        'Action, Sci-Fi',
        'Action, Adventure, Fantasy'
    ],
    'plot': [
        'Batman fights against the Joker in Gotham City',
        'A thief enters dreams to plant ideas',
        'Astronauts travel through a wormhole to save humanity',
        'A computer programmer discovers the truth about reality',
        'A marine explores an alien planet and joins their tribe'
    ]
}

# Create DataFrame
movies_df = pd.DataFrame(movies_data)

# Combine genres and plot for content-based filtering
movies_df['content'] = movies_df['genres'] + ' ' + movies_df['plot']

# TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer converts text into numerical vectors by considering:
#
# 1. Term Frequency (TF): How often a word appears in a document
#    Example: In "I love this movie", "love" has TF=1, "movie" has TF=1
#
# 2. Inverse Document Frequency (IDF): How unique a word is across all documents
#    - Words appearing in many documents (like "the", "is") get lower weights
#    - Rare words get higher weights
#    Example: If "movie" appears in all documents (lower IDF), "love" in only two (higher IDF)
#
# 3. TF-IDF Score = TF × IDF
#    - Common words in a document but rare across all documents get higher scores
#    - Common words that appear everywhere get lower scores
#
# This allows us to:
# - Convert movie descriptions into numerical vectors
# - Give more weight to distinctive terms that better characterize each movie
# - Ignore common words that don't help differentiate between movies

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['content'])

# Calculate cosine similarity between movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(movie_title, cosine_sim=cosine_sim, df=movies_df):
    """
    Get movie recommendations based on content similarity
    """
    # Get the index of the movie
    idx = df[df['title'] == movie_title].index[0]
    
    # Get similarity scores for all movies
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top 3 similar movies (excluding the movie itself)
    sim_scores = sim_scores[1:4]
    
    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return recommended movies
    return df['title'].iloc[movie_indices]

# Example usage
if __name__ == "__main__":
    movie_title = "The Dark Knight"
    print(f"\nRecommendations for '{movie_title}':")
    recommendations = get_recommendations(movie_title)
    for i, movie in enumerate(recommendations, 1):
        print(f"{i}. {movie}")
