# Collaborative Filtering Recommendation System

A deep learning-based movie recommendation system using collaborative filtering with PyTorch. This project implements matrix factorization with neural networks to predict user preferences and generate personalized recommendations.

## Author
**Edze Worla Alex**

## Features

- **Neural Collaborative Filtering**: Uses deep learning with embedding layers
- **Matrix Factorization**: Learns latent factors for users and items
- **Bias Terms**: Incorporates user, item, and global biases
- **Dropout Regularization**: Prevents overfitting
- **Batch Training**: Efficient training with mini-batches
- **Model Persistence**: Save and load trained models
- **Evaluation Metrics**: RMSE and MSE for performance assessment
- **Top-N Recommendations**: Generate personalized item recommendations

## Technical Architecture

### Model Components
1. **User Embeddings**: Dense vector representations of users
2. **Item Embeddings**: Dense vector representations of items
3. **Bias Terms**: User-specific, item-specific, and global biases
4. **Dropout Layers**: Regularization to prevent overfitting

### Loss Function
- Mean Squared Error (MSE) between predicted and actual ratings

### Optimization
- Adam optimizer with customizable learning rate

## Requirements

```
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/edze-worla-alex/recommendation-system.git
cd recommendation-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from collaborative_filtering import CollaborativeFilteringRecommender
import pandas as pd

# Load your ratings data (userId, movieId, rating columns required)
ratings_df = pd.read_csv('ratings.csv')

# Initialize the recommender
recommender = CollaborativeFilteringRecommender(
    embedding_dim=128,
    learning_rate=0.001,
    num_epochs=100,
    batch_size=512
)

# Train the model
recommender.train(ratings_df)

# Get recommendations for a user
user_id = 123
recommendations = recommender.recommend_items(
    user_id=user_id,
    n_recommendations=10,
    ratings_df=ratings_df
)

print(f"Top 10 recommendations for user {user_id}: {recommendations}")
```

### Advanced Usage

```python
# Custom training with validation split
recommender.train(ratings_df, test_size=0.2, random_state=42)

# Plot training history
recommender.plot_training_history()

# Make predictions for specific user-item pairs
user_ids = [1, 2, 3]
item_ids = [100, 200, 300]
predictions = recommender.predict(user_ids, item_ids)

# Save the trained model
recommender.save_model('my_model.pth')

# Load a saved model
new_recommender = CollaborativeFilteringRecommender()
new_recommender.load_model('my_model.pth')
```

## Data Format

Your ratings data should be in CSV format with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| userId | int | Unique user identifier |
| movieId | int | Unique item/movie identifier |
| rating | float | Rating value (e.g., 1-5 scale) |

Example:
```csv
userId,movieId,rating
1,31,2.5
1,1029,3.0
1,1061,3.0
1,1129,2.0
```

## Model Parameters

### CollaborativeFilteringRecommender Parameters

- `embedding_dim` (int, default=128): Dimension of user and item embeddings
- `learning_rate` (float, default=0.001): Learning rate for optimization
- `num_epochs` (int, default=100): Number of training epochs
- `batch_size` (int, default=512): Batch size for training
- `device` (str, default=auto): Device to use ('cuda' or 'cpu')

### MatrixFactorization Parameters

- `num_users` (int): Number of unique users
- `num_items` (int): Number of unique items
- `embedding_dim` (int, default=128): Embedding dimension
- `dropout` (float, default=0.2): Dropout rate for regularization

## Performance Metrics

The model is evaluated using:

1. **RMSE (Root Mean Square Error)**: Lower values indicate better performance
2. **MSE (Mean Square Error)**: Lower values indicate better performance

## Model Architecture Details

```
Input: User ID, Item ID
    ↓
User Embedding (embedding_dim) ← User Bias (1)
Item Embedding (embedding_dim) ← Item Bias (1)
    ↓
Dropout Regularization
    ↓
Dot Product + Biases + Global Bias
    ↓
Output: Predicted Rating
```

## Example Results

```
=== Training Results ===
Users: 943, Items: 1682
Epoch [10/50], Train Loss: 0.8234, Val Loss: 0.8456
Epoch [20/50], Train Loss: 0.7123, Val Loss: 0.7234
...
Training completed!

=== Evaluation ===
Test RMSE: 0.8456
Test MSE: 0.7150

=== Recommendations for User 196 ===
1. Movie 1467
2. Movie 1201
3. Movie 1189
4. Movie 1122
5. Movie 814
```

## File Structure

```
recommendation-system/
├── collaborative_filtering.py    # Main implementation
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── examples/
│   ├── basic_example.py        # Basic usage example
│   └── advanced_example.py     # Advanced features demo
├── data/
│   └── sample_ratings.csv      # Sample data file
└── models/
    └── pretrained_model.pth    # Saved model example
```

## Customization

### Custom Loss Functions

```python
class CustomMatrixFactorization(MatrixFactorization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add custom layers or modifications
        
    def forward(self, user_ids, item_ids):
        # Custom forward pass
        return super().forward(user_ids, item_ids)
```

### Custom Data Preprocessing

```python
def preprocess_ratings(df):
    # Custom preprocessing logic
    # E.g., normalize ratings, handle missing values
    df['rating'] = (df['rating'] - df['rating'].mean()) / df['rating'].std()
    return df
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size` or `embedding_dim`
2. **Poor recommendations**: Increase `embedding_dim` or `num_epochs`
3. **Overfitting**: Increase `dropout` rate or reduce model complexity
4. **Slow training**: Use GPU acceleration or reduce data size

### Performance Tips

1. Use GPU acceleration for large datasets
2. Experiment with different embedding dimensions
3. Tune learning rate and batch size
4. Use early stopping to prevent overfitting
5. Consider data normalization for better convergence

## Extensions

Possible extensions to this project:

1. **Content-based filtering**: Incorporate item features
2. **Hybrid models**: Combine collaborative and content-based filtering
3. **Deep neural networks**: Add more hidden layers
4. **Attention mechanisms**: Focus on relevant user-item interactions
5. **Sequential recommendations**: Consider temporal patterns
6. **Multi-task learning**: Predict multiple objectives simultaneously

## References

1. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems.
2. He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural collaborative filtering.
3. Sedhain, S., Menon, A. K., Sanner, S., & Xie, L. (2015). AutoRec: Autoencoders meet collaborative filtering.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

**Edze Worla Alex**
- GitHub: [@edze-worla-alex]
- Email: edze.worla@gmail.com

---

*This project was created as part of a data science portfolio to demonstrate expertise in recommendation systems and deep learning.*