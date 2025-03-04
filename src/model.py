import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_model(df, model_path='models/sales_forecast_model.pkl'):
    """
    Trains a Linear Regression model and saves it to a file.
    """
    # Features and target
    X = df[['Profit']]
    y = df['Sales']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred) if len(y_test) >= 2 else None

    print(f"Model trained successfully!")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2}")

    # Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    return model
