import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix


data = pd.read_csv('dataset_med.csv')


print(data.head())


print(data.info())  
print(data.describe())  

data.fillna(data.mean(), inplace=True)  
data.fillna(data.mode().iloc[0], inplace=True)  


label_encoder = LabelEncoder()
categorical_columns = ['gender', 'country', 'cancer_stage', 'family_history', 
                      'smoking_status', 'hypertension', 'asthma', 'cirrhosis', 'other_cancer', 'treatment_type']

for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])


features = ['age', 'gender', 'country', 'bmi', 'cholesterol_level', 
            'hypertension', 'asthma', 'cirrhosis', 'other_cancer', 
            'cancer_stage', 'smoking_status', 'treatment_type']
target = 'survived'

X = data[features]
y = data[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{classification_report}')
print(f'ROC-AUC Score: {roc_auc}')


conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')
