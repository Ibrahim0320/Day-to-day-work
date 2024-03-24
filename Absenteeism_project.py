# Abesnteeism Data Analysis
# Using employee data to predict absenteeism from work

'''
import numpy as np
import pandas as pd

raw_data= pd.read_csv('Relevant CSV Files/Absenteeism_data.csv')

pd.options.display.max_columns = None
pd.options.display.max_rows= None

data= raw_data.copy()
data= raw_data.drop(['ID'], axis=1)

reason_columns= pd.get_dummies(data['Reason for Absence'], drop_first= True)

reason_columns['check']= reason_columns.sum(axis=1)

reason_columns['check'].sum(axis=0)
reason_columns['check'].unique()

reason_columns= reason_columns.drop(['check'], axis=1)

data= data.drop(['Reason for Absence'], axis= 1)



reason_type1= reason_columns.loc[:, 1: 14].max(axis=1)
reason_type2= reason_columns.loc[:, 15: 17].max(axis=1)
reason_type3= reason_columns.loc[:, 18: 21].max(axis=1)
reason_type4= reason_columns.loc[:, 22: ].max(axis=1)

data= pd.concat([data, reason_type1, reason_type2, reason_type3, reason_type4], axis=1)



column_names= ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']


data.columns= column_names

data_names_reordered= ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 
                          'Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours']

data= data[data_names_reordered]

data_reason_mod= data.copy()


data_reason_mod['Date']= pd.to_datetime(data_reason_mod['Date'], format= '%d/%m/%Y')



list_months= []
for i in range(data_reason_mod['Date'].shape[0]):
    list_months.append(data_reason_mod['Date'][i].month)

data_reason_mod['Month Value'] = list_months



def date_to_weekday(date_value):
    return date_value.weekday()

data_reason_mod['Day of the Week']= data_reason_mod['Date'].apply(date_to_weekday)

data_reason_mod= data_reason_mod.drop(['Date'], axis= 1)


new_column_names= ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value', 'Day of the Week',
       'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
       'Pets', 'Absenteeism Time in Hours']

data_reason_mod= data_reason_mod[new_column_names]

data_reason_date_mod= data_reason_mod.copy()

data_reason_date_mod['Education'].unique()
data_reason_date_mod['Education'].value_counts()
data_reason_date_mod['Education'] = data_reason_date_mod['Education'].map({1:0, 2:1, 3:1, 4:1})
data_reason_date_mod['Education'].unique()
data_reason_date_mod['Education'].value_counts()

data_preprocessed = data_reason_date_mod.copy()
data_preprocessed.head()


data_preprocessed.to_csv('Absenteeism_preprocessed.csv', index=False)

'''


# Machine Learning Model

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


pd.options.display.max_columns = None
pd.options.display.max_rows= None

data= pd.read_csv('Relevant CSV Files/Absenteeism_preprocessed.csv')

targets= np.where(data['Absenteeism Time in Hours'] > data['Absenteeism Time in Hours'].median(), 1, 0)
data['Excessive Absenteeism'] = targets

data_with_targets= data.drop(['Absenteeism Time in Hours'], axis=1)

#print(data_with_targets.shape)
unscaled_inputs= data_with_targets.iloc[:, :-1]

absenteeism_scaler= StandardScaler()

absenteeism_scaler.fit(unscaled_inputs)
scaled_inputs= absenteeism_scaler.transform(unscaled_inputs)

x_train, x_test, y_train, y_test= train_test_split(scaled_inputs, targets, train_size= 0.8, random_state=20)

#print(x_train.shape, y_train.shape)
#print(x_test.shape, y_test.shape)



# Logistic Regression

from sklearn.preprocessing import StandardScaler
absenteeism_scaler = StandardScaler()


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# create the Custom Scaler class

class CustomScaler(BaseEstimator,TransformerMixin): 
    
    # init or what information we need to declare a CustomScaler object
    # and what is calculated/declared as we do
    
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        
        # scaler is nothing but a Standard Scaler object
        self.scaler = StandardScaler(copy=copy,with_mean=with_mean,with_std=with_std)
        # with some columns 'twist'
        self.columns = columns
        self.mean_ = None
        self.var_ = None
# the fit method, which, again based on StandardScale
    
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    # the transform method which does the actual scaling
    def transform(self, X, y=None, copy=None):
        # record the initial order of the columns
        init_col_order = X.columns
        # scale all features that you chose when creating the instance of the class
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        # declare a variable containing all information that was not scaled
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        # return a data frame which contains all scaled features and all 'not scaled' features
        # use the original order (that you recorded in the beginning)
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


unscaled_inputs.columns.values
columns_to_omit = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Education']

columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]


absenteeism_scaler = CustomScaler(columns_to_scale)
absenteeism_scaler.fit(unscaled_inputs)

scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)

#print(scaled_inputs.head(), scaled_inputs.shape )

from sklearn.model_selection import train_test_split

train_test_split(scaled_inputs, targets)

x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, #train_size = 0.8, 
                                                    test_size = 0.2, random_state = 20)

#print(x_train.shape, y_train.shape)
#print(x_test.shape, y_test.shape)


from sklearn.linear_model import LogisticRegression
from sklearn import metrics

reg= LogisticRegression()


reg.fit(x_train, y_train)
#print(reg.score(x_test, y_test))

# Manually check the accuracy, aka build the score function

model_outputs= reg.predict(x_train)
model_outputs == y_train

np.sum(model_outputs==y_train)
model_outputs.shape
np.sum(model_outputs==y_train) / model_outputs.shape[0]


# Finding intercepts and coefficients

reg.intercept_, reg.coef_
feature_name= unscaled_inputs.columns.values
summary_table= pd.DataFrame(columns=['Feature Name'], data= feature_name)
summary_table['Coefficient'] = np.transpose(reg.coef_)
#print(summary_table)

summary_table.index= summary_table.index + 1
summary_table.loc[0]= ['Intercept', reg.intercept_[0]]
summary_table= summary_table.sort_index()

#Interpreting the coefficients

summary_table['Odds Ratio']= np.exp(summary_table.Coefficient)
summary_table.sort_values('Odds Ratio', ascending=False)
# A feature is not particularly important if its Coef is ≈0 and the odds ratio is ≈1

predicted_proba= reg.predict_proba(x_test)
predicted_proba[:, :-1]



# Save the model

import pickle

# model file
#with open('model', 'wb') as file:
#    pickle.dump(reg, file)
# scaler file
#with open('scaler','wb') as file:
#    pickle.dump(absenteeism_scaler, file)

