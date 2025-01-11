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


df.dropna(subset=['price'],inplace=True)
df['mileage'].fillna(df['mileage'].median(),inplace=True)
df=pd.get_dummies(df,columns=['make','fuel','transmission'],drop_first=True)
df.drop(columns=['name','description','exterior_color','interior_color'],inplace=True)
print(df['engine'].unique())


sns.histplot(df['price'],bins=30, kde=True)
plt.show()
