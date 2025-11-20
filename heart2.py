import numpy as np  # Numerical computations
import pandas as pd  # Data manipulation
from sklearn.model_selection import train_test_split
from sklearn import tree  # Tree package for decision tree model
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import seaborn as sns
import matplotlib.pyplot as plt

data = 'heart.csv'
df = pd.read_csv(data)
print(df.head()) #prints first 5 rows


print(df.shape) #prints rows and columns


print(df.isna().sum()) #checking null values

print(df.dtypes) #prints the datatypes

df.info()  #prints number of entries(rows), column names and datatypes, non-null counts, memory usage    



#data visualization part
df['age'].hist(grid=True, bins=15)
plt.title('Age distribution')  # corrected spelling
plt.xlabel('Age')
plt.ylabel('Number of Patients')
plt.show()




# Using kdeplot (clean and modern)
sns.kdeplot(df[df['sex'] == 1]['age'], label='Male', fill=True)
sns.kdeplot(df[df['sex'] == 0]['age'], label='Female', fill=True)

plt.title('Density Plot of Age by Sex')
plt.xlabel('Age')
plt.ylabel('Density')

plt.show()




df['trestbps'].hist(bins=10, grid=True)
plt.title('Resting Blood Pressure Distribution')
plt.xlabel('Resting Blood Pressure (mm Hg)')
plt.ylabel('Number of Patients')
plt.show()


sns.kdeplot(df['trestbps'], fill=True)
plt.title('Resting Blood pressure desnity plot')
plt.xlabel('Resting Blood Pressure')
plt.ylabel('Density')
plt.show()



import plotly.express as px

fig = px.histogram(df, x='trestbps', nbins=15, title='Resting Blood Pressure Distribution')
fig.show()




fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))  # Optional: add figsize for better layout

# Boxplot for 'chol' (Cholesterol)
sns.boxplot(y='chol', data=df, ax=axes[0])  # Changed x → y for vertical orientation

# Boxplot for 'oldpeak'
sns.boxplot(y='oldpeak', data=df, ax=axes[1])  # Changed x → y for vertical orientation

# Add titles
axes[0].set_title('Cholesterol (chol)')
axes[1].set_title('ST Depression (oldpeak)')

plt.tight_layout()
plt.show()


fig = px.box(df, y='chol', title='Cholesterol Boxplot')
fig.add_box(y=df['oldpeak'], name='Oldpeak')
fig.show()
fig.write_html('plotly_grapgh.html')



import pandas as pd

# Use a relative or absolute path to the CSV file
data = pd.read_csv('heart.csv')  # If in the same folder




from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()
report = AV.AutoViz('heart.csv')  #AutoViz: A Python library that can automatically visualize data based on a given DataFrame



import pandas as pd
from pycaret.classification import *
df = pd.read_csv('heart.csv')
clf = setup(data = df,target = 'target',session_id=123)
best_model = compare_models()
print(best_model)