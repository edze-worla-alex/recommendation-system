from collaborative_filtering import *
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

filepath = os.path.join(MODEL_DIR, 'collaborative_filtering_model.pth')

def main():
    """Main execution function"""
    
    print("=== Collaborative Filtering Recommendation System ===")
    
    # Generate sample data (in practice, load your own dataset)
    print("Generating sample data...")
    ratings_df = generate_sample_data()
    print(f"Generated {len(ratings_df)} ratings for {ratings_df['userId'].nunique()} users "
          f"and {ratings_df['movieId'].nunique()} movies")
    
    # Split data for evaluation
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    
    # Initialize and train the recommender
    recommender = CollaborativeFilteringRecommender(
        embedding_dim=64,
        learning_rate=0.001,
        num_epochs=50,
        batch_size=256
    )
    
    # Train the model
    recommender.train(train_df)
    
    # Plot training history
    recommender.plot_training_history()
    
    # Evaluate the model
    evaluate_model(recommender, test_df)
    
    # Make recommendations for a sample user
    sample_user = ratings_df['userId'].iloc[0]
    recommendations = recommender.recommend_items(
        user_id=sample_user,
        n_recommendations=10,
        ratings_df=ratings_df
    )
    
    print(f"\nTop 10 recommendations for User {sample_user}:")
    for i, movie_id in enumerate(recommendations, 1):
        print(f"{i}. Movie {movie_id}")
    
    # Save the model
    recommender.save_model(filepath)
    
    print("\nRecommendation system training completed!")

if __name__ == "__main__":
    main()