# Heart Attack Prediction

This project predicts the likelihood of a heart attack based on clinical health parameters using machine learning.  
It includes data preprocessing, exploratory data analysis (EDA), automated visualization, and multiple ML models for comparison.

## 📂 Project Contents
- `heart.csv` — Dataset containing medical features.
- `HeartAttackPrediction.ipynb` — Full notebook with EDA, visualization, model training, and evaluation.
- `heart2.py` — Script version of the model pipeline.
- AutoViz and PyCaret pipelines for automated insights and model benchmarking.

## 🧠 Problem Definition
Given patient medical parameters such as age, cholesterol, blood pressure, and ECG results, the goal is to classify whether a patient is at risk for a heart attack (`target` = 1) or not (`target` = 0).

## 🛠 Tools & Libraries
- Python 3.x  
- Pandas, NumPy  
- Matplotlib, Seaborn, Plotly  
- Scikit-learn  
- AutoViz  
- PyCaret  

## 📊 Features Used
Key clinical indicators:
- age  
- sex  
- cp  
- trestbps  
- chol  
- fbs  
- restecg  
- thalach  
- exang  
- oldpeak  
- slope  
- ca  
- thal  
- target (label)

## 🚀 Model Training Workflow
1. Load and inspect dataset  
2. Perform EDA: distributions, correlations, KDE plots, boxplots  
3. Visualize trends using Matplotlib, Seaborn, Plotly  
4. Use AutoViz for automated visualization  
5. Use PyCaret to compare ML models (Logistic Regression, Random Forest, XGBoost, etc.)  
6. Select the best model and evaluate accuracy + confusion matrix  

## 📈 Results
- PyCaret automatically identifies the highest-performing model.
- Visualizations highlight major risk factors.
- Provides a quick baseline ML system for heart attack prediction.

## 📦 How to Run
### Install dependencies
```sh
pip install pandas numpy matplotlib seaborn plotly pycaret autoviz scikit-learn
````

### Run the notebook

```sh
jupyter notebook HeartAttackPrediction.ipynb
```

### Run the script

```sh
python heart2.py
```

## 👤 Author

**Sampath Magapu**

## 📝 License — MIT License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2025 Sampath Magapu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
