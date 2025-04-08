from xgboost import XGBRegressor
import joblib
import os

def train_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("✅ Model training complete.")
    return model

def save_model(model, path='models/model.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"💾 Model saved to {path}")
