import pandas as pd
import matplotlib.pyplot as plt  

import os  
  
# 获取当前工作路径  
current_working_directory = os.getcwd()  
  
# 打印当前工作路径  
print("当前工作路径是:", current_working_directory)

data = pd.read_csv('train.csv')
print(data.columns)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


y = data.readmitted

base_features = [c for c in data.columns if c != "readmitted"]

X = data[base_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)
row_to_show = 5
data_for_prediction = val_X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)


my_model.predict_proba(data_for_prediction_array)

import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# Calculate Shap values
""" shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()
shap.plots.force(explainer.expected_value[1],shap_values[1],data_for_prediction)
 """

# calculate shap values. This is what we will plot.
# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
shap_values = explainer.shap_values(val_X)
print(shap_values.shape)
#shap.plots.force(explainer.expected_value[1],shap_values[1],data_for_prediction)

# Make plot. Index of [1] is explained in text below.
""" shap.summary_plot(shap_values[1], val_X)
plt.savefig("shap_summary.png") """