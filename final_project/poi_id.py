#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import pickle
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import recall_score, accuracy_score, precision_score, \
classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion


sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

'''financial only
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', \
'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',\
 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', \
 'restricted_stock', 'director_fees']
email only
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


# Exploring dataset

print "count of people in dataset: ", len(data_dict)
poicount = sum(p['poi']==1 for p in data_dict.values())
print "count of poi is: ", poicount
print "count of non-poi is: ", (len(data_dict) - poicount)

# print everyone's name
#for person in data_dict.keys():
#    print person

print "\nNaN count of feature: "
count = [0 for i in range(len(features_list))]
for i, person in enumerate(data_dict.values()):
    for j, feature in enumerate(features_list):
        if person[feature] == 'NaN':
            count[j] += 1
for i, feature in enumerate(features_list):
    print feature, count[i]
print "\n"

''' # visualize all features in features_list
for i in range(1, len(features_list)):
    features_box = features_list[0], features_list[i]    
    data_box = featureFormat(data_dict, features_box)

    #create a dataframe from data_box
    df = pd.DataFrame(data_box, columns = ['poi', features_list[i]])
    
    #plot using poi as the group
    sns.boxplot(df[features_list[i]], groupby = df['poi'])
    sns.plt.show()
'''

outliers = []
for key in data_dict:
    val = data_dict[key]['bonus']
    #val = data_dict[key]['salary']
    if val == 'NaN':
        continue;    
    outliers.append((key,int(val)))
#print top 3 bonuses    
print (sorted(outliers,key=lambda x:x[1],reverse=True)[:3]), "\n"


### Task 2: Remove outliers
outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
for outlier in outliers:
    data_dict.pop(outlier, 0)

   
### Task 3: Create new feature(s)
for person in data_dict.values():
    person['to_poi_message_ratio'] = 0
    person['from_poi_message_ratio'] = 0
    if float(person['from_messages']) > 0:
        person['to_poi_message_ratio'] = \
        float(person['from_this_person_to_poi'])/float(person['from_messages'])
        person['from_poi_message_ratio'] = \
        float(person['from_poi_to_this_person'])/float(person['to_messages'])

'''#print new poi ratios   
for key, person in data_dict.iteritems():
    print key, ": ", person['to_poi_message_ratio']
    print key, ": ", person['from_poi_message_ratio']
'''    
features_list.extend(['to_poi_message_ratio', 'from_poi_message_ratio']) 
   
### Store to my_dataset for easy export below.
my_dataset = data_dict


### Extract features and labels from dataset for local testing, scale features

#data = featureFormat(my_dataset, features_list, sort_keys = True)
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


'''
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# feature selection selectkbest
selector = SelectKBest(f_classif, k=8).fit(features, labels)
scores = selector.scores_
unsorted_pairs = zip(features_list[1:], scores)
sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
print "\nSelectKBest features:"
kbest_features_list = dict(sorted_pairs[:8]).keys()

print kbest_features_list, "\n"
kbest_features_list.insert(0, 'poi')
'''

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# 1. Using scaled Features
'''features_train, features_test, labels_train, labels_test = \
train_test_split(features, labels, test_size=0.3, random_state=42)

# Kneighbors Classifier
clf_knc = KNeighborsClassifier(algorithm='auto', leaf_size=30, \
metric='manhattan', metric_params=None, n_neighbors=4, p=2, weights='distance')

# Apply the classifier to Scaled Data
clf_knc.fit(features_train, labels_train)
y_pred = clf_knc.predict(features_test)

print "Classification report: Scaled Features" 
print classification_report(labels_test, y_pred)
print ' '     

## Pipeline
pipeline = Pipeline([('normalization', scaler),
                     ('classifier', clf_knc)
])
clf = pipeline
'''

pca = PCA()
kbest = SelectKBest(f_classif)
gnb = GaussianNB()

combined_features = FeatureUnion([("pca", pca), ("kbest", kbest)])

pipeline = Pipeline([("features", combined_features), ("gnb", gnb)])
param_grid_neigh = dict(features__pca__n_components=[2,3,4,5,10,15,20],
                  features__kbest__k=[2,3,4,5,10,15,20])    
clf = GridSearchCV(pipeline, param_grid=param_grid_neigh, verbose=3, \
    scoring = 'f1')
clf.fit(features,labels)

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

test_classifier(clf, my_dataset, features_list)

dump_classifier_and_data(clf, my_dataset, features_list)