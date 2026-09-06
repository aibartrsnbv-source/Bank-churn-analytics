import pandas as pd
from sqlalchemy import create_engine

# Загрузил данные с risk score
df = pd.read_csv('data/clean_churn_data_with_risk.csv')

# Подключмл к PostgreSQL 
DB_USER = 'postgres'
DB_PASSWORD = 'Aibar2006'
DB_HOST = 'localhost'
DB_PORT = '5433'
DB_NAME = 'bank_churn_project'

engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

df.to_sql('clients', engine, if_exists='replace', index=False)
print(f"Загружено {len(df)} строк в таблицу clients")