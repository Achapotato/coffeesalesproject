import pandas as pd
import joblib
from datetime import datetime

# load model
model = joblib.load('model/random_forest_model.joblib')

#
# input data
input_data = {
    'transaction_qty': [2, 3, 5],
    'weekday': [1, 4, 6],
    'transaction_hour': ['07 AM', '02 PM', '10 AM']  # 字符串形式需转换
}

# build DataFrame
input_df = pd.DataFrame(input_data)

# '07 AM' converted to int format（int）
input_df['transaction_hour'] = pd.to_datetime(input_df['transaction_hour'], format='%I %p').dt.hour

# choose features
X = input_df[['transaction_qty', 'weekday', 'transaction_hour']]

# predict
predictions = model.predict(X)

# print prediction result
print("Predicted Sales Amount:")
print(predictions)

