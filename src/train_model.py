import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib


# Step 1: Load data
df = pd.read_csv('data/preprocessed_dataset.csv')

# Step 2: convert transaction_hour as hour
from dateutil import parser

def convert_hour_to_float(time_str):
    try:
        return parser.parse(time_str).hour
    except Exception as e:
        print(f"⛔ Error converting time: {time_str}, {e}")
        return np.nan

df['transaction_hour'] = df['transaction_hour'].apply(convert_hour_to_float)


# Step 3: check missing value
print("🔍 Missing values before dropna:")
print(df[['transaction_qty', 'weekday', 'transaction_hour', 'sales_amount']].isnull().sum())

# Step 4: delete missing value
df = df.dropna(subset=['transaction_qty', 'weekday', 'transaction_hour', 'sales_amount'])

# Step 5: check again
if df.empty:
    print("❌ After dropna, dataframe is empty. Please check input CSV.")
else:
    print("✅ Dataset shape after cleaning:", df.shape)
    print("🔍 Sample rows:\n", df[['transaction_qty', 'weekday', 'transaction_hour', 'sales_amount']].head())

    # Step 6: Prepare X and y
    X = df[['transaction_qty', 'weekday', 'transaction_hour']]
    y = df['sales_amount']

    # Step 7: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 8: Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 9: Evaluate
    y_pred = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    print(f"📊 RMSE of Random Forest: {rmse:.3f}")

    joblib.dump(model, 'model/random_forest_model.joblib')

    # Step 10: Save model
    joblib.dump(model, 'model/random_forest_model.pkl')
    print("✅ Model saved to model/random_forest_model.pkl")

df[['transaction_qty', 'weekday', 'transaction_hour', 'sales_amount']].to_csv('data/processed_dataset.csv', index=False)


