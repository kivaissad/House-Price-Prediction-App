# House-Price-Prediction-App

This is a House Price Prediction App made by using basic Machine Learning algorithms and developed using Streamlit.

## Description of the project
1. The dataset is taken from a Kaggle Competition. The training dataset is stored in `train.csv`.
2. The description of the labels and columns in the dataset is given in `data_description.txt`
3. The necessary libraries and their dependencies are mentioned in `requirements.txt`. Install them using the given command: 
   ```bash
   pip install -r requirements.txt
   ```
4. Loading the data, visualisation and the preprocessing is done in the `EDA.ipynb` (Exploratory Data Analysis Notebook). The preprocessing is done using numpy, pandas, seaborn and matplotlib. The cleaned data is saved in `X_train.csv`.
5. The important features are selected and engineered in `Feature_Engineering.ipynb` and `Feature_Selection.ipynb` and the final data is saved in `Final_X_Train.csv`. The features are selected using LassoCV from SelectFromModel module of Scikit-learn.
6. The final model is built in `model.py` and displayed using streamlit. The application also uses shap for visualization the correlation between different parameters. The user input is a slider of different parameters normalised between 0 and 1. The output is the resulting price of the house for that input.
