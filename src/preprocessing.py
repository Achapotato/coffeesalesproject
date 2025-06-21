import pandas as pd

# data read
df = pd.read_csv("data/coffeerawdataset1.csv")

# remove duplicate value
df_cleaned = df.drop_duplicates()

# convert transaction_time to datetime
df_cleaned['transaction_time'] = pd.to_datetime(
    df_cleaned['transaction_time'],
    format='%I:%M:%S %p',
    errors='coerce'
)
# extract weekday from transaction date
df_cleaned["weekday"] = pd.to_datetime(df_cleaned["transaction_date"]).dt.weekday


# add new column：keep hour + AM/PM
df_cleaned['transaction_hour'] = df_cleaned['transaction_time'].dt.strftime('%I %p')

# calculate sales amount
df_cleaned['sales_amount'] = df_cleaned['transaction_qty'] * df_cleaned['unit_price']

# present data shape
print(f"✅ Cleaned data shape: {df_cleaned.shape}")

# save processed data
df_cleaned.to_csv("data/preprocessed_dataset.csv", index=False)
print("✅ Preprocessed dataset saved to data/preprocessed_dataset.csv")

