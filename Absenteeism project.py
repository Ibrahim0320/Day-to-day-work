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


pd.options.display.max_columns = None
pd.options.display.max_rows= None

data= pd.read_csv('Absenteeism_preprocessed.csv')
print(data.head())