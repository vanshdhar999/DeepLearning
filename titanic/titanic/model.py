import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

X = train_data.drop(['Survived', 'Ticket', 'PassengerId', 'Name', 'Cabin'], axis=1)
y = train_data['Survived']


X_train, X_val, y_train, y_valid = train_test_split(X, y, train_size= 0.8, test_size=0.2, random_state=0)

#Data processing 

categorical_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and 
                        X_train[cname].dtype == "object"]


numerical_cols = [cname for cname in 
                  X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

print("Categorical Columns and numerical columns->",categorical_cols, numerical_cols)

# Apply one-hot encoder to each column of categorical data 

my_cols = categorical_cols + numerical_cols

X_full_train = X_train[my_cols].copy()
X_full_valid = X_val[my_cols].copy()

print(" X_train Before processing- >\n",X_full_train.isnull().sum())

# Preprocessing for numerical data 

numerical_transformer = SimpleImputer(strategy='constant')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model = RandomForestClassifier(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])
#Fitting the data 
# Pipeline method simultaneously pre processes the data and fits the model. 
my_pipeline.fit(X_full_train, y_train)

preds = my_pipeline.predict(X_full_valid) # The pipeline defined by us will first pre-proces the validation data and then make the predictions. 

score = mean_absolute_error(y_valid, preds)

X_test = test_data[my_cols].copy()

preds_test = my_pipeline.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data['PassengerId'],
                       'Survived': preds_test})
output.to_csv('titanic_predictions1.csv', index=False)