import pandas as pd
import joblib

def predict():
    model = joblib.load("models/model.pkl")
    df_test = pd.read_csv("data/blind_test_data.csv")
    selected_features = [
        'feature_2',
        'feature_13',
        'feature_9',
        'feature_11',
        'feature_18'
    ]

    X_test = df_test[selected_features]
    preds = model.predict(X_test)
    pd.DataFrame({"target_pred": preds}).to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")
    return "Predictions saved to predictions.csv"
