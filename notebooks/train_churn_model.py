import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

# Загрузил очищенные данные
df = pd.read_csv('data/clean_churn_data.csv')

# Кодировал категориальные признаки
le_country = LabelEncoder()
le_gender = LabelEncoder()
df['country_encoded'] = le_country.fit_transform(df['country'])
df['gender_encoded'] = le_gender.fit_transform(df['gender'])

# Убираем tenure_group и исходные текстовые столбцы
df_model = df.drop(columns=['country', 'gender', 'tenure_group', 'customer_id'])

# Разделил на признаки и целевую переменную
X = df_model.drop(columns=['churn'])
y = df_model['churn']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Обучил модель
model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, scale_pos_weight=3.9, random_state=42)
model.fit(X_train, y_train)

# Оценка
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("\n", classification_report(y_test, y_pred))

# Признаки которые сильнее всего влияют на отток
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nВажность признаков:\n", importance)

# Сохранил предсказания вероятности риска для каждого клиента 
df['churn_risk_score'] = model.predict_proba(df_model.drop(columns=['churn']))[:, 1]
df.to_csv('data/clean_churn_data_with_risk.csv', index=False)
print("\nСохранено с risk score в data/clean_churn_data_with_risk.csv")