import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

class MovieLensDataset(Dataset):
    """Custom Dataset for MovieLens data"""
    
    def __init__(self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray):
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.ratings = torch.FloatTensor(ratings)
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

class MatrixFactorization(nn.Module):
    """Neural Collaborative Filtering Model"""
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 128, dropout: float = 0.2):
        super(MatrixFactorization, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Bias terms
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        
        # Global bias
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # Get embeddings
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        
        # Apply dropout
        user_embeds = self.dropout(user_embeds)
        item_embeds = self.dropout(item_embeds)
        
        # Compute dot product
        dot_product = torch.sum(user_embeds * item_embeds, dim=1)
        
        # Add bias terms
        user_bias = self.user_bias(user_ids).squeeze()
        item_bias = self.item_bias(item_ids).squeeze()
        
        # Final prediction
        prediction = dot_product + user_bias + item_bias + self.global_bias
        
        return prediction

class CollaborativeFilteringRecommender:
    """Main Recommendation System Class"""
    
    def __init__(self, embedding_dim: int = 128, learning_rate: float = 0.001, 
                 num_epochs: int = 100, batch_size: int = 512, device: str = None):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.user_encoder = None
        self.item_encoder = None
        self.train_losses = []
        self.val_losses = []
        
    def prepare_data(self, ratings_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare and encode user-item data"""
        
        # Create user and item encoders
        unique_users = ratings_df['userId'].unique()
        unique_items = ratings_df['movieId'].unique()
        
        self.user_encoder = {user: idx for idx, user in enumerate(unique_users)}
        self.item_encoder = {item: idx for idx, item in enumerate(unique_items)}
        
        # Reverse encoders for decoding
        self.user_decoder = {idx: user for user, idx in self.user_encoder.items()}
        self.item_decoder = {idx: item for item, idx in self.item_encoder.items()}
        
        # Encode users and items
        users = ratings_df['userId'].map(self.user_encoder).values
        items = ratings_df['movieId'].map(self.item_encoder).values
        ratings = ratings_df['rating'].values
        
        return users, items, ratings
    
    def train(self, ratings_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """Train the collaborative filtering model"""
        
        print("Preparing data...")
        users, items, ratings = self.prepare_data(ratings_df)
        
        # Split data
        users_train, users_val, items_train, items_val, ratings_train, ratings_val = \
            train_test_split(users, items, ratings, test_size=test_size, random_state=random_state)
        
        # Create datasets and dataloaders
        train_dataset = MovieLensDataset(users_train, items_train, ratings_train)
        val_dataset = MovieLensDataset(users_val, items_val, ratings_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize model
        num_users = len(self.user_encoder)
        num_items = len(self.item_encoder)
        
        self.model = MatrixFactorization(num_users, num_items, self.embedding_dim)
        self.model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        print(f"Training model on {self.device}...")
        print(f"Users: {num_users}, Items: {num_items}")
        
        # Training loop
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_users, batch_items, batch_ratings in train_loader:
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)
                batch_ratings = batch_ratings.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(batch_users, batch_items)
                loss = criterion(predictions, batch_ratings)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_users, batch_items, batch_ratings in val_loader:
                    batch_users = batch_users.to(self.device)
                    batch_items = batch_items.to(self.device)
                    batch_ratings = batch_ratings.to(self.device)
                    
                    predictions = self.model(batch_users, batch_items)
                    loss = criterion(predictions, batch_ratings)
                    val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], '
                      f'Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}')
        
        print("Training completed!")
    
    def predict(self, user_ids: List[int], item_ids: List[int]) -> np.ndarray:
        """Make predictions for user-item pairs"""
        
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Encode user and item IDs
        encoded_users = [self.user_encoder.get(uid, 0) for uid in user_ids]
        encoded_items = [self.item_encoder.get(iid, 0) for iid in item_ids]
        
        users_tensor = torch.LongTensor(encoded_users).to(self.device)
        items_tensor = torch.LongTensor(encoded_items).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(users_tensor, items_tensor)
            return predictions.cpu().numpy()
    
    def recommend_items(self, user_id: int, n_recommendations: int = 10, 
                       exclude_seen: bool = True, ratings_df: pd.DataFrame = None) -> List[int]:
        """Recommend top-N items for a user"""
        
        if user_id not in self.user_encoder:
            print(f"User {user_id} not found in training data")
            return []
        
        # Get all items
        all_items = list(self.item_decoder.keys())
        
        # Exclude items the user has already rated
        if exclude_seen and ratings_df is not None:
            user_items = set(ratings_df[ratings_df['userId'] == user_id]['movieId'].values)
            item_candidates = [self.item_encoder[item] for item in self.item_decoder.values() 
                             if item not in user_items]
        else:
            item_candidates = all_items
        
        if not item_candidates:
            return []
        
        # Predict ratings for all candidate items
        user_ids = [user_id] * len(item_candidates)
        candidate_items = [self.item_decoder[item] for item in item_candidates]
        
        predictions = self.predict(user_ids, candidate_items)
        
        # Sort by predicted rating
        item_scores = list(zip(candidate_items, predictions))
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-N recommendations
        recommendations = [item for item, score in item_scores[:n_recommendations]]
        return recommendations
    
    def plot_training_history(self):
        """Plot training and validation loss"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save!")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'user_decoder': self.user_decoder,
            'item_decoder': self.item_decoder,
            'embedding_dim': self.embedding_dim
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.user_encoder = checkpoint['user_encoder']
        self.item_encoder = checkpoint['item_encoder']
        self.user_decoder = checkpoint['user_decoder']
        self.item_decoder = checkpoint['item_decoder']
        
        num_users = len(self.user_encoder)
        num_items = len(self.item_encoder)
        embedding_dim = checkpoint['embedding_dim']
        
        self.model = MatrixFactorization(num_users, num_items, embedding_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {filepath}")

def generate_sample_data(num_users: int = 1000, num_movies: int = 500, 
                        num_ratings: int = 50000) -> pd.DataFrame:
    """Generate sample movie ratings data"""
    
    np.random.seed(42)
    
    # Generate random user-movie interactions
    users = np.random.randint(1, num_users + 1, num_ratings)
    movies = np.random.randint(1, num_movies + 1, num_ratings)
    
    # Generate ratings with some bias (popular movies get higher ratings)
    ratings = []
    for movie in movies:
        if movie <= 50:  # Popular movies (1-50)
            rating = np.random.choice([3, 4, 5], p=[0.2, 0.4, 0.4])
        elif movie <= 200:  # Average movies (51-200)
            rating = np.random.choice([2, 3, 4, 5], p=[0.2, 0.3, 0.3, 0.2])
        else:  # Less popular movies (201+)
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.2, 0.3, 0.3, 0.1, 0.1])
        ratings.append(rating)
    
    # Create DataFrame and remove duplicates
    df = pd.DataFrame({
        'userId': users,
        'movieId': movies,
        'rating': ratings
    })
    
    # Remove duplicate user-movie pairs, keeping the last rating
    df = df.drop_duplicates(subset=['userId', 'movieId'], keep='last')
    
    return df

def evaluate_model(recommender, test_df: pd.DataFrame):
    """Evaluate the model's performance"""
    
    print("Evaluating model...")
    
    # Predict ratings for test set
    test_users = test_df['userId'].tolist()
    test_items = test_df['movieId'].tolist()
    test_ratings = test_df['rating'].values
    
    predictions = recommender.predict(test_users, test_items)
    
    # Calculate metrics
    mse = mean_squared_error(test_ratings, predictions)
    rmse = np.sqrt(mse)
    
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MSE: {mse:.4f}")
    
    return rmse, mse

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
    recommender.save_model('collaborative_filtering_model.pth')
    
    print("\nRecommendation system training completed!")

if __name__ == "__main__":
    main()