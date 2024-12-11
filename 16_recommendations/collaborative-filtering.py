import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFiltering:
    def __init__(self):
        self.user_ratings = None
        self.similarity_matrix = None

    def fit(self, ratings_matrix):
        """
        Fit the collaborative filtering model with a user-item ratings matrix
        
        Parameters:
        ratings_matrix (DataFrame): Matrix with users as rows, items as columns, and ratings as values
        """
        self.user_ratings = ratings_matrix
        # Calculate user similarity matrix using cosine similarity
        self.similarity_matrix = pd.DataFrame(
            cosine_similarity(ratings_matrix.fillna(0)),
            index=ratings_matrix.index,
            columns=ratings_matrix.index
        )

    def recommend(self, user_id, n_recommendations=5):
        """
        Generate recommendations for a specific user
        
        Parameters:
        user_id: The user to generate recommendations for
        n_recommendations (int): Number of recommendations to generate
        
        Returns:
        List of recommended items
        """
        if user_id not in self.user_ratings.index:
            return "User not found"

        # Get user's ratings
        user_ratings = self.user_ratings.loc[user_id]
        
        # Find items the user hasn't rated
        unrated_items = user_ratings[user_ratings.isna()].index
        
        # Calculate predicted ratings for unrated items
        predictions = {}
        
        for item in unrated_items:
            # Get all ratings for this item
            item_ratings = self.user_ratings[item]
            
            # Get similarities to other users
            similarities = self.similarity_matrix[user_id]
            
            # Calculate weighted average rating
            numerator = 0
            denominator = 0
            
            for other_user in self.user_ratings.index:
                if other_user != user_id and not pd.isna(item_ratings[other_user]):
                    similarity = similarities[other_user]
                    rating = item_ratings[other_user]
                    
                    numerator += similarity * rating
                    denominator += similarity
            
            if denominator > 0:
                predictions[item] = numerator / denominator
        
        # Sort predictions and return top n recommendations
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_predictions[:n_recommendations]

# Example usage
if __name__ == "__main__":
    # Create a sample ratings matrix
    # Rows are users, columns are items, values are ratings (1-5)
    data = {
        'Item1': [5, 3, np.nan, 4, 4],
        'Item2': [3, 1, 2, 3, np.nan],
        'Item3': [4, 3, 4, 3, 5],
        'Item4': [3, 3, 1, 5, 4],
        'Item5': [1, 5, 5, 2, 1]
    }
    ratings_df = pd.DataFrame(data, index=['User1', 'User2', 'User3', 'User4', 'User5'])
    
    # Initialize and fit the recommender
    recommender = CollaborativeFiltering()
    recommender.fit(ratings_df)
    
    # Get recommendations for User3
    recommendations = recommender.recommend('User3', n_recommendations=2)
    
    print("\nSample Ratings Matrix:")
    print(ratings_df)
    print("\nRecommendations for User3:")
    for item, predicted_rating in recommendations:
        print(f"{item}: Predicted rating = {predicted_rating:.2f}")
