import pandas as pd

# read raw data
df = pd.read_csv('data/coffeerawdataset1.csv')

# show row number
print(f"Original data shape: {df.shape}")

# delete duplicate value
df_cleaned = df.drop_duplicates()

# represent row number
print(f"Cleaned data shape: {df_cleaned.shape}")

# save preprocessed data
df_cleaned.to_csv('data/preprocessed_dataset.csv', index=False)

print("Preprocessed dataset saved to data/preprocessed_dataset.csv")
