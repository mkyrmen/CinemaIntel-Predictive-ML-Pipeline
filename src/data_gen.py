import pandas as pd
import numpy as np
import os

# Ensure directory exists
os.makedirs("data", exist_ok=True)

def generate_movie_data(n_samples=1500):
    np.random.seed(42)
    
    data = {
        'Budget_M': np.random.gamma(shape=2, scale=40, size=n_samples),
        'Genre': np.random.choice(['Action', 'Drama', 'Comedy', 'Sci-Fi', 'Horror'], n_samples),
        'Runtime_Min': np.random.normal(110, 25, n_samples),
        'Critic_Score': np.random.beta(a=5, b=2, size=n_samples) * 100,
        'Release_Month': np.random.randint(1, 13, n_samples),
        'Star_Power_Index': np.random.uniform(1, 10, n_samples)
    }

    df = pd.DataFrame(data)

    # Complex success logic: Interaction between Genre, Budget, and Score
    # Example: Horror is profitable even at low budgets; Sci-Fi needs high scores to offset cost.
    success_score = (df['Critic_Score'] * 0.4) + (df['Star_Power_Index'] * 1.5)
    success_score += np.where(df['Genre'] == 'Horror', 15, 0)
    success_score -= (df['Budget_M'] * 0.05)
    
    # Target: 1 = Hit, 0 = Flop (Top 40% are hits)
    df['Success_Label'] = (success_score > np.percentile(success_score, 60)).astype(int)

    df.to_csv("Z:/Project/Movie_Predictor/raw_data/raw_movies.csv", index=False)
    print("✅ Synthetic dataset created: data/raw_movies.csv")

if __name__ == "__main__":
    generate_movie_data()