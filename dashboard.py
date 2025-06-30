# Streamlit Dashboard with Filters, Visualizations, and Random Forest Forecast

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta
import os

# Load preprocessed dataset for filtering and visualization
file_path = 'data/preprocessed_dataset.csv'
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    st.success("Preprocessed dataset loaded successfully.")
else:
    st.error("Preprocessed dataset not found.")
    st.stop()

# Convert date column
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

# Sidebar filters
st.sidebar.header("Filters")
product_categories = ['All'] + list(df['product_category'].unique())
locations = ['All'] + list(df['store_location'].unique())
date_range = [df['transaction_date'].min(), df['transaction_date'].max()]

selected_category = st.sidebar.selectbox("Select Product Category", product_categories)
selected_location = st.sidebar.selectbox("Select Store Location", locations)
selected_start_date = st.sidebar.date_input("Start Date", date_range[0])
selected_end_date = st.sidebar.date_input("End Date", date_range[1])


# Apply filters
filtered_df = df.copy()

# Filter by product category if not 'All'
if selected_category != 'All':
    filtered_df = filtered_df[filtered_df['product_category'] == selected_category]

# Filter by store location if not 'All'
if selected_location != 'All':
    filtered_df = filtered_df[filtered_df['store_location'] == selected_location]

# Filter by date range
filtered_df = filtered_df[
    (filtered_df['transaction_date'] >= pd.to_datetime(selected_start_date)) &
    (filtered_df['transaction_date'] <= pd.to_datetime(selected_end_date))
]


# ---- summary metrics ----
st.subheader("Summary Metrics (for selected filters)")

total_qty   = int(filtered_df['transaction_qty'].sum())
total_sales = float(filtered_df['sales_amount'].sum())

col1, col2 = st.columns(2)
col1.metric("Total Quantity Sold", f"{total_qty:,}")
col2.metric("Total Sales Amount", f"${total_sales:,.2f}")


# --- Visualization 1: Monthly Sales Trend ---
st.subheader("Monthly Sales Trend")
monthly_sales = df.set_index('transaction_date').resample('MS')['transaction_qty'].sum()
monthly_sales = monthly_sales.asfreq('MS', fill_value=0)  # fill missing months with 0
fig1, ax1 = plt.subplots()
ax1.plot(monthly_sales.index, monthly_sales.values)
ax1.set_title("Monthly Sales Trend")
ax1.set_xlabel("Month")
ax1.set_ylabel("Quantity")
st.pyplot(fig1)

# --- Visualization 2: Sales by Hour of Day ---
st.subheader("Hourly Sales Distribution")
if 'transaction_hour' in df.columns:
    if df['transaction_hour'].dtype == object:
        def parse_hour(h):
            try:
                return datetime.strptime(h, '%I %p').hour
            except:
                return -1
        df['transaction_hour'] = df['transaction_hour'].apply(parse_hour)
    hourly_sales = df.groupby('transaction_hour')['transaction_qty'].sum().sort_index()
    fig2, ax2 = plt.subplots()
    ax2.bar(hourly_sales.index, hourly_sales.values)
    ax2.set_title("Sales by Hour")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Quantity")
    ax2.set_xticks(range(0, 24))
    ax2.set_xticklabels([f"{h:02d}" for h in range(24)], rotation=45)
    st.pyplot(fig2)

# --- Forecast with Random Forest ---
st.subheader("Demand Forecast Using Random Forest")

forecast_path = 'data/processed_dataset.csv'
model_path = 'model/random_forest_model.joblib'

if os.path.exists(model_path) and os.path.exists(forecast_path):
    model = joblib.load(model_path)
    df_forecast = pd.read_csv(forecast_path)

    if not df_forecast.empty:
        try:
            if hasattr(model, 'feature_names_in_'):
                df_forecast = df_forecast[model.feature_names_in_]

            prediction = model.predict(df_forecast)

            # Keep only 30-day forecast
            forecast_days = 30
            prediction = prediction[:forecast_days]
            start_date = df['transaction_date'].max() + timedelta(days=1)
            forecast_dates = pd.date_range(start=start_date, periods=forecast_days, freq='D')

            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Predicted Quantity': prediction
            }).set_index('Date')

            st.line_chart(forecast_df['Predicted Quantity'])

        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.warning("Processed dataset is empty.")
else:
    st.warning("Forecast model or dataset not found.")







