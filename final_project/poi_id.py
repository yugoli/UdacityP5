#!/usr/bin/python

import sys
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import decomposition, preprocessing


sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
'''
The features in the data fall into three major types, namely financial 
features, email features and POI labels. 

financial features: 
['salary', 'deferral_payments', 'total_payments', 'loan_advances', 
'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 
 'restricted_stock', 'director_fees'] (all units are in US dollars) 
 
 email features: 
 ['to_messages', 'email_address', 'from_poi_to_this_person', 
 'from_messages', 'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi']
 (units are generally number of emails messages; notable exception is 
 ‘email_address’, which is a text string) 
 
 POI label: 
 [‘poi’] (boolean, represented as integer)
'''
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# removed restricted_stock_deferred, director_fees only 1 label

'''financial only
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', \
'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',\
 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', \
 'restricted_stock', 'director_fees']
'''
'''email only
features_list = ['poi', 'to_messages', 'from_poi_to_this_person', \
 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
'''
#all default features
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', \
'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',\
 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', \
 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', \
 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# EDA

print "count of people in dataset: ", len(data_dict)
poicount = sum(p['poi']==1 for p in data_dict.values())
print "count of poi is: ", poicount
print "count of non-poi is: ", (len(data_dict) - poicount)

for person in data_dict.keys():
    print person

print "\nNaN count of feature: "
count = [0 for i in range(len(features_list))]
for i, person in enumerate(data_dict.values()):
    for j, feature in enumerate(features_list):
        if person[feature] == 'NaN':
            count[j] += 1
for i, feature in enumerate(features_list):
    print feature, count[i]

from pprint import pprint
bonus_outliers = []
salary_outliers = []
for key in data_dict:
    val = data_dict[key]['bonus']
    val1 = data_dict[key]['salary']
    if val == 'NaN':
        continue
    bonus_outliers.append((key,int(val)))
    salary_outliers.append((key,int(val1)))

pprint(sorted(bonus_outliers,key=lambda x:x[1],reverse=True)[:2])
pprint(sorted(salary_outliers,key=lambda x:x[1],reverse=True)[:2])
print ""

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
    
''' # visualizing the features
for i in range(1, len(features_list)):
    features_box = features_list[0], features_list[i]    
    data_box = featureFormat(data_dict, features_box)

    #create a dataframe from data_box
    df = pd.DataFrame(data_box, columns = ['poi', features_list[i]])
    
    #plot using poi as the group
    sns.boxplot(df[features_list[i]], groupby = df['poi'])
    sns.plt.show()
 '''
   
### Task 3: Create new feature(s), scale financial features
for person in data_dict.values():
    person['to_poi_message_ratio'] = 0
    person['from_poi_message_ratio'] = 0
    if float(person['from_messages']) > 0:
        person['to_poi_message_ratio'] = float(person['from_this_person_to_poi'])/float(person['from_messages'])
        person['from_poi_message_ratio'] = float(person['from_poi_to_this_person'])/float(person['to_messages'])

'''#print new poi ratios   
for key, person in data_dict.iteritems():
    print key, ": ", person['to_poi_message_ratio']
    print key, ": ", person['from_poi_message_ratio']
'''    
features_list.extend(['to_poi_message_ratio', 'from_poi_message_ratio']) 
   
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# feature selection selectkbest
selector = SelectKBest(f_classif, k=5).fit(features, labels)
#print np.sort(selector.scores_)[::-1]
print "\nSelectKBest features"
for i in selector.get_support(indices=True):
    print features_list[i+1], " ", selector.scores_[i]
#print [features_list[i+1] for i in selector.get_support(indices=True)]
#print [selector.scores_[i] for i in selector.get_support(indices=True)]




# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)