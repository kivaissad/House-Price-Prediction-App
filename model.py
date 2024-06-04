import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
pd.pandas.set_option('display.max_columns', None)

st.write("""
# Boston-like House Price Prediction App

This app predicts the **House Prices** from the *Input Dataset*
""")
st.write('---')

# Load the cleaned up dataset
dataset=pd.read_csv('X_train.csv')
X=dataset.drop(['Id', 'SalePrice'], axis=1)
Y=dataset[['SalePrice']]

print(X)
# Check which features which matter the most and choose them as user input features
feature_sel_model=SelectFromModel(Lasso(alpha=0.005, random_state=0))
feature_sel_model.fit(X, Y)
selected_features=X.columns[(feature_sel_model.get_support())]
X=X[selected_features]
X.rename(columns={'1stFlrSF':'FirstFlrSF'}, inplace=True)
# [MSSubClass', 'MSZoning', 'Neighborhood', 'OverallQual', 'YearRemodAdd', 'RoofStyle', 'BsmtQual', 'BsmtExposure', 'HeatingQC', 'CentralAir', 'FirstFlrSF', 'GrLivArea', 'BsmtFullBath', 'KitchenQual', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageCars', 'PavedDrive', 'SaleCondition']

# Sidebar
# Header to specify input parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    MSSubClass=st.sidebar.slider('MSSubClass', X.MSSubClass.min(), X.MSSubClass.max(), X.MSSubClass.mean())
    MSZoning=st.sidebar.slider('MSZoning', X.MSZoning.min(), X.MSZoning.max(), X.MSZoning.mean())
    Neighborhood=st.sidebar.slider('Neighborhood', X.Neighborhood.min(), X.Neighborhood.max(), X.Neighborhood.mean())
    OverallQual=st.sidebar.slider('OverallQual', X.OverallQual.min(), X.OverallQual.max(), X.OverallQual.mean())
    YearRemodAdd=st.sidebar.slider('YearRemodAdd', X.YearRemodAdd.min(), X.YearRemodAdd.max(), X.YearRemodAdd.mean())
    RoofStyle=st.sidebar.slider('RoofStyle', X.RoofStyle.min(), X.RoofStyle.max(), X.RoofStyle.mean())
    BsmtQual=st.sidebar.slider('BsmtQual', X.BsmtQual.min(), X.BsmtQual.max(), X.BsmtQual.mean())
    BsmtExposure=st.sidebar.slider('BsmtExposure', X.BsmtExposure.min(), X.BsmtExposure.max(), X.BsmtExposure.mean())
    HeatingQC=st.sidebar.slider('HeatingQC', X.HeatingQC.min(), X.HeatingQC.max(), X.HeatingQC.mean())
    CentralAir=st.sidebar.slider('CentralAir', X.CentralAir.min(), X.CentralAir.max(), X.CentralAir.mean())
    FirstFlrSF=st.sidebar.slider('FirstFlrSF', X.FirstFlrSF.min(), X.FirstFlrSF.max(), X.FirstFlrSF.mean())
    GrLivArea=st.sidebar.slider('GrLivArea', X.GrLivArea.min(), X.GrLivArea.max(), X.GrLivArea.mean())
    BsmtFullBath=st.sidebar.slider('BsmtFullBath', X.BsmtFullBath.min(), X.BsmtFullBath.max(), X.BsmtFullBath.mean())
    KitchenQual=st.sidebar.slider('KitchenQual', X.KitchenQual.min(), X.KitchenQual.max(), X.KitchenQual.mean())
    Fireplaces=st.sidebar.slider('Fireplaces', X.Fireplaces.min(), X.Fireplaces.max(), X.Fireplaces.mean())
    FireplaceQu=st.sidebar.slider('FireplaceQu', X.FireplaceQu.min(), X.FireplaceQu.max(), X.FireplaceQu.mean())
    GarageType=st.sidebar.slider('GarageType', X.GarageType.min(), X.GarageType.max(), X.GarageType.mean())
    GarageFinish=st.sidebar.slider('GarageFinish', X.GarageFinish.min(), X.GarageFinish.max(), X.GarageFinish.mean())
    GarageCars=st.sidebar.slider('GarageCars', X.GarageCars.min(), X.GarageCars.max(), X.GarageCars.mean())
    PavedDrive=st.sidebar.slider('PavedDrive', X.PavedDrive.min(), X.PavedDrive.max(), X.PavedDrive.mean())
    SaleCondition=st.sidebar.slider('SaleCondition', X.SaleCondition.min(), X.SaleCondition.max(), X.SaleCondition.mean())

    data={'MSSubClass': MSSubClass,
          'MSZoning': MSZoning,
          'Neighborhood': Neighborhood,
          'OverallQual': OverallQual,
          'YearRemodAdd': YearRemodAdd,
          'RoofStyle': RoofStyle,
          'BsmtQual': BsmtQual,
          'BsmtExposure': BsmtExposure,
          'HeatingQC': HeatingQC,
          'CentralAir': CentralAir,
          'FirstFlrSF': FirstFlrSF,
          'GrLivArea': GrLivArea,
          'BsmtFullBath': BsmtFullBath,
          'KitchenQual': KitchenQual,
          'Fireplaces': Fireplaces,
          'FireplaceQu': FireplaceQu,
          'GarageType': GarageType,
          'GarageFinish': GarageFinish,
          'GarageCars': GarageCars,
          'PavedDrive': PavedDrive,
          'SaleCondition': SaleCondition}
    features=pd.DataFrame(data, index=[0])
    return features

user_input=user_input_features()

# Main Panel
# Print the input parameters
st.header('Specified Input Parameters')
st.write(user_input)
st.write('---')

# Build Regressor Model
model=RandomForestRegressor()
model.fit(X, Y)

# Apply Model to make predictions
prediction=model.predict(user_input)

st.header('Prediction of SalePrice')
st.write(np.floor(np.exp(prediction)))
st.write('---')

# Visualising the predictions using SHAP values
st.set_option('deprecation.showPyplotGlobalUse', False)
explain=shap.TreeExplainer(model)
shap_values=explain.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (BarPlot)')
shap.summary_plot(shap_values, X, plot_type='bar')
st.pyplot(bbox_inches='tight')