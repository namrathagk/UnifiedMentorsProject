import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('dataset.csv')
print(df.head())
print(df.describe())


print(df.isnull().sum())


print(df['gender'].unique())
print(df['country'].unique())
print(df['cancer_stage'].unique())
print(df['family_history'].unique())
print(df['smoking_status'].unique())
print(df['treatment_type'].unique())
print(df['survived'].unique())


sns.countplot(x='survived', data=df)
plt.title('Survival Count')
plt.show()
sns.histplot(df['price'],bins=30, kde=True)
plt.show()
