import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

os.makedirs("models", exist_ok=True)

def train_pipeline():
    # 1. Load and Encode
    df = pd.read_csv("Z:/Project/Movie_Predictor/raw_data/raw_movies.csv")
    df = pd.get_dummies(df, columns=['Genre']) # One-Hot Encoding

    X = df.drop('Success_Label', axis=1)
    y = df['Success_Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Hyperparameter Tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }

    print("⚙️ Tuning Random Forest via Grid Search...")
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # 3. Save Best Model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, "models/optimized_movie_model.pkl")
    
    # 4. Preliminary Metrics
    probs = best_model.predict_proba(X_test)[:, 1]
    print(f"🏆 Best Params: {grid_search.best_params_}")
    print(f"📈 AUC-ROC Score: {roc_auc_score(y_test, probs):.2f}")
    
    return X_test, y_test

if __name__ == "__main__":
    train_pipeline()