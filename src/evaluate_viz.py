import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import pandas as pd
import os

os.makedirs("reports", exist_ok=True)

def evaluate():
    # Load model and data
    model = joblib.load("models/optimized_movie_model.pkl")
    df = pd.read_csv("Z:/Project/Movie_Predictor/raw_data/raw_movies.csv")
    df_encoded = pd.get_dummies(df, columns=['Genre'])
    
    # Sample last 300 rows as 'new' test data
    X_test = df_encoded.drop('Success_Label', axis=1).tail(300)
    y_test = df_encoded['Success_Label'].tail(300)
    y_pred = model.predict(X_test)

    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Purples')
    plt.title('Confusion Matrix: Hit Prediction Accuracy')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig("Z:/Project/Movie_Predictor/reports/confusion_matrix.png")

    # 2. Feature Importance
    importance = pd.Series(model.feature_importances_, index=X_test.columns).sort_values()
    plt.figure(figsize=(10, 6))
    importance.plot(kind='barh', color='midnightblue')
    plt.title('Predictive Power of Movie Features')
    plt.tight_layout()
    plt.savefig("Z:/Project/Movie_Predictor/reports/feature_importance.png")
    
    print("📊 Evaluation charts saved to /reports")
    print("\nFinal Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate()