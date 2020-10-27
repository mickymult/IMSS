
# Commented out IPython magic to ensure Python compatibility.
# Canada Post Corporation; Calgary Mail Processing Plant. Oct. 2020
# Machine Learning Model to predict total parcels processed per shift
# Based on induction volumes, staffing & scan tunnel read rate

# Import the dependencies, that will make this program a little easier to write. 
# Importing the machine learning library sklearn, matplotlib, numpy, and pandas.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
#import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("C:/Users/user/pyproj/ship_sorter/imss_performance.csv")
data = np.array(data)
#df = pd.read_csv("C:/Users/user/pyproj/ship_sorter/imss_performance.csv")

#plt.figure(figsize=(50,25))
#fig, ax = plt.subplots()
#sns.heatmap(df.corr(),annot =True)
#st.pyplot(fig)


X = data[:, :-1]
y = data[:, -1]
y = y.astype('int')
X = X.astype('int')
kwargs = dict(test_size=0.2, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, **kwargs)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

#print(regressor.intercept_)

#Print the coefecients/weights for each feature/column of our model
#print(regressor.coef_)

#print our parcel processing predictions on our test data
y_pred = regressor.predict(X_test)
#print(X_test)
whole_y = y_pred.astype(int)
print (whole_y)

def main():
    st.title("CMPP Machine Learning")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Procesed Volume Prediction</h2>
    </div>
    """

    ind_vol = st.number_input('Induction Volume', min_value=2000, max_value=26000, value=20000)
    q1ind = st.selectbox('Induction Staff Q1', [0,1,2,3,4,5,6,7,8,9,10])
    q1run = st.selectbox('Runout Staff Q1', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
    q1oth = st.selectbox('Other Staff Q1', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

    q2ind = st.selectbox('Induction Staff Q2', [0,1,2,3,4,5,6,7,8,9,10])
    q2run = st.selectbox('Runout Staff Q2', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
    q2oth = st.selectbox('Other Staff Q2', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

    q3ind = st.selectbox('Induction Staff Q3', [0,1,2,3,4,5,6,7,8,9,10])
    q3run = st.selectbox('Runout Staff Q3', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
    q3oth = st.selectbox('Other Staff Q3', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

    q4ind = st.selectbox('Induction Staff Q4', [0,1,2,3,4,5,6,7,8,9,10])
    q4run = st.selectbox('Runout Staff Q4', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
    q4oth = st.selectbox('Other Staff Q4', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

    scanrr = st.number_input('Scan Tunnel Read Rate', min_value=70.5, max_value=99.9, value=98.5)
    output = regressor.predict([[ind_vol,q1ind,q1run,q1oth,q2ind,q2run,q2oth,q3ind,q3run,q3oth,q4ind,q4run,q4oth,scanrr]])
    int_output = output.astype(int)

    if st.button("Predict"):
        
        st.success('Predicted Volume Processed {}'.format(int_output))

#df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#df

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

if __name__=='__main__':
    main()
