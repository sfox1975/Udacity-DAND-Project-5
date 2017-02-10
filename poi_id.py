'''
Please refer to the accompanying iPython Notebook (poi_id.ipynb)
and the final report for details on the EDA and justification for
much of the coding modifications, additions and deletions given
below. The intent of this file is for project grading purposes
'''


#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Convert data dictionary into a pandas dataframe, for ease of manipulation

df = pd.DataFrame.from_records(list(data_dict.values()))
employees = pd.Series(list(data_dict.keys()))
df.set_index(employees, inplace = True)

df.replace('NaN', 0, inplace = True)

# Remove outliers

df = df.drop(['TOTAL', 'THE TRAVEL AGENCY IN THE PARK'])

# Manually correct erroneous data for Sanjay Bhatnagar

df.loc['BHATNAGAR SANJAY','director_fees'] = 0
df.loc['BHATNAGAR SANJAY','exercised_stock_options'] = 15456290
df.loc['BHATNAGAR SANJAY','expenses'] = 137864
df.loc['BHATNAGAR SANJAY','other'] = 0
df.loc['BHATNAGAR SANJAY','restricted_stock'] = 2604490
df.loc['BHATNAGAR SANJAY','restricted_stock_deferred'] = -2604490
df.loc['BHATNAGAR SANJAY','total_payments'] = 137864
df.loc['BHATNAGAR SANJAY','total_stock_value'] = 15456290

# Manually correct erroneous data for Robert Belfer

df.loc['BELFER ROBERT','deferral_payments'] = 0
df.loc['BELFER ROBERT','deferred_income'] = -102500
df.loc['BELFER ROBERT','director_fees'] = 102500
df.loc['BELFER ROBERT','exercised_stock_options'] = 0
df.loc['BELFER ROBERT','expenses'] = 3285
df.loc['BELFER ROBERT','restricted_stock'] = 44093
df.loc['BELFER ROBERT','restricted_stock_deferred'] = -44093
df.loc['BELFER ROBERT','total_payments'] = 3285
df.loc['BELFER ROBERT','total_stock_value'] = 0

# Encode new features

df.replace(0, np.nan, inplace = True)

df['TP_TSV_ratio'] = df['total_payments'] / df['total_stock_value']

df['deferred_total_ratio'] = df['deferred_income']+df['restricted_stock_deferred']/(
    df['total_payments']+df['total_stock_value'])

df['to_poi_ratio'] = df['from_this_person_to_poi'] / df['from_messages']

df['from_poi_ratio'] = df['from_poi_to_this_person'] / df['to_messages']


# Replace np.nan with NaN (for compatibility with feature_format.py)

df.replace(np.nan, 'NaN', inplace = True)

# create a dictionary from the dataframe
df_dict = df.to_dict('index')


# Store to my_dataset for easy export below.
my_dataset = df_dict

# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".

features_list = ['poi','salary', 'deferral_payments', 'total_payments', 
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 
'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
'long_term_incentive', 'restricted_stock', 'director_fees','to_messages',
'from_poi_to_this_person', 'from_messages','from_this_person_to_poi', 
'shared_receipt_with_poi','to_poi_ratio','from_poi_ratio','TP_TSV_ratio',
'deferred_total_ratio']


# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# Final Tuned algorithm (see iPython notebook and final report for
# extensive details on the algorithm optimization process)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(splitter="random",
                             max_depth=4,
                             min_samples_split=10,
                             max_leaf_nodes=8,
                             class_weight="balanced",
                             random_state=42)

clf.fit(features,labels)

# Dump the classifier, dataset, and features_list for testing via
# tester.py

dump_classifier_and_data(clf, my_dataset, features_list)