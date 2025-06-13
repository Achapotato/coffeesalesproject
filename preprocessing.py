import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv("coffeerawdataset1.csv")  # adjust the path if needed

# Step 2: Remove duplicate rows
df_cleaned = df.drop_duplicates()

# Step 3: Convert 'transaction_time' to hour only (e.g., "7:06:11 AM" -> 7)
df_cleaned['transaction_hour'] = pd.to_datetime(df_cleaned['transaction_time']).dt.hour

# Step 4: Drop the original 'transaction_time' column if no longer needed
df_cleaned = df_cleaned.drop(columns=['transaction_time'])

# Optional: Preview the cleaned DataFrame
print(df_cleaned.head())
