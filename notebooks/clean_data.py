import pandas as pd
df = pd.read_csv('data/raw/Bank Customer Churn Prediction.csv')

# Обзор данных
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())

#Создал и сохранил производные признаки
df['balance_to_salary_ratio'] = df['balance'] / df['estimated_salary'].replace(0, 1)
df['is_high_value_client'] = (df['balance'] > df['balance'].quantile(0.75)).astype(int)
df['tenure_group'] = pd.cut(df['tenure'], bins=[-1, 2, 5, 20], labels=['0-2', '3-5', '6+'])

print(df.head())
print(df['tenure_group'].value_counts())

df.to_csv('data/clean_churn_data.csv', index=False)
print("Сохранено в data/clean_churn_data.csv")