from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import os

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"📊 Evaluation Metrics:\nRMSE: {rmse:.2f}\nR² Score: {r2:.2f}")
    return preds, rmse, r2

def predict_and_prepare_submission(model, X_test, test_store, output_path='data/rossmann_submission.csv'):
    preds = model.predict(X_test)
    
    preds = np.clip(preds, a_min=0, a_max=None)
    
    test_store = test_store.copy()
    test_store['PredictedSales'] = preds

    submission = test_store[['Id', 'PredictedSales']]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"✅ Submission file saved as {output_path}")
