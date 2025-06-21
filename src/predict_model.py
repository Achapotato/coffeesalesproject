import pandas as pd
import joblib
from datetime import datetime

# 加载训练好的模型
model = joblib.load('model/random_forest_model.joblib')

# 模拟输入数据（你也可以从CSV文件读取，比如 pd.read_csv('data/future_input.csv')）
# 示例数据结构需与训练时相同
input_data = {
    'transaction_qty': [2, 3, 5],
    'weekday': [1, 4, 6],
    'transaction_hour': ['07 AM', '02 PM', '10 AM']  # 字符串形式需转换
}

# 创建 DataFrame
input_df = pd.DataFrame(input_data)

# 将 '07 AM' 等字符串格式转换为数字小时（int）
input_df['transaction_hour'] = pd.to_datetime(input_df['transaction_hour'], format='%I %p').dt.hour

# 选择与训练时相同的特征列
X = input_df[['transaction_qty', 'weekday', 'transaction_hour']]

# 执行预测
predictions = model.predict(X)

# 打印预测结果
print("Predicted Sales Amount:")
print(predictions)

