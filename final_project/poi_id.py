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
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import BernoulliRBM
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, recall_score, accuracy_score, \
precision_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from numpy import asarray


sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

'''financial only
features_list = ['poi','salary', 'deferral_payments', 'total_payments', \
'loan_advances', \
'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',\
 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', \
 'restricted_stock', 'director_fees']
email only
features_list = ['poi', 'to_messages', 'from_poi_to_this_person', \
 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
'''

#all default features
features_list = [
                "poi",
                "from_messages",
                "from_poi_to_this_person",
                "from_this_person_to_poi",
                "shared_receipt_with_poi",
                "to_messages",
                "bonus",
                "deferral_payments",
                "deferred_income",
                "director_fees",
                "exercised_stock_options",
                "expenses",
                "loan_advances",
                "long_term_incentive",
                "other",
                "restricted_stock",
                "restricted_stock_deferred",
                "salary",
                "total_payments",
                "total_stock_value"                
                ]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

def eda(data_dict, features_list, fillnan=True):
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
                if fillnan:
                    person[feature] = 0
    for i, feature in enumerate(features_list):
        print feature, count[i]
    print "\n"
    
    
        
        
    ''' 
    ##### visualize all features in features_list
    for i in range(1, len(features_list)):
        features_box = features_list[0], features_list[i]    
        data_box = featureFormat(data_dict, features_box)
    
        #create a dataframe from data_box
        df = pd.DataFrame(data_box, columns = ['poi', features_list[i]])
        
        #plot using poi as the group
        sns.boxplot(df[features_list[i]], groupby = df['poi'])
        sns.plt.show()
    '''
    
    top_bonus = []
    for key in data_dict:
        val = data_dict[key]['bonus']
        #val = data_dict[key]['salary']
        if val == 'NaN':
            continue;    
        top_bonus.append((key,int(val)))
    #print top 3 bonuses    
    print (sorted(top_bonus,key=lambda x:x[1],reverse=True)[:3]), "\n"

def remove_outliers(data_dict):
    ### Task 2: Remove outliers
    outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
    for outlier in outliers:
        data_dict.pop(outlier, 0)
    return data_dict
def correct_records(data_dict):
    #Fix two records
    data_dict['BELFER ROBERT'] = {'bonus': 'NaN',
                              'deferral_payments': 'NaN',
                              'deferred_income': -102500,
                              'director_fees': 102500,
                              'email_address': 'NaN',
                              'exercised_stock_options': 'NaN',
                              'expenses': 3285,
                              'from_messages': 'NaN',
                              'from_poi_to_this_person': 'NaN',
                              'from_this_person_to_poi': 'NaN',
                              'loan_advances': 'NaN',
                              'long_term_incentive': 'NaN',
                              'other': 'NaN',
                              'poi': False,
                              'restricted_stock': -44093,
                              'restricted_stock_deferred': 44093,
                              'salary': 'NaN',
                              'shared_receipt_with_poi': 'NaN',
                              'to_messages': 'NaN',
                              'total_payments': 3285,
                              'total_stock_value': 'NaN'}

    data_dict['BHATNAGAR SANJAY'] = {'bonus': 'NaN',
                                 'deferral_payments': 'NaN',
                                 'deferred_income': 'NaN',
                                 'director_fees': 'NaN',
                                 'email_address': 'sanjay.bhatnagar@enron.com',
                                 'exercised_stock_options': 15456290,
                                 'expenses': 137864,
                                 'from_messages': 29,
                                 'from_poi_to_this_person': 0,
                                 'from_this_person_to_poi': 1,
                                 'loan_advances': 'NaN',
                                 'long_term_incentive': 'NaN',
                                 'other': 'NaN',
                                 'poi': False,
                                 'restricted_stock': 2604490,
                                 'restricted_stock_deferred': -2604490,
                                 'salary': 'NaN',
                                 'shared_receipt_with_poi': 463,
                                 'to_messages': 523,
                                 'total_payments': 137864,
                                 'total_stock_value': 15456290} 
    return data_dict
    
def create_new_features(data_dict, features_list):   
    ### Task 3: Create new feature(s)
    for person in data_dict.values():
        #email features        
        person['to_poi_message_ratio'] = 0
        person['from_poi_message_ratio'] = 0
        if float(person['from_messages']) > 0:
            person['to_poi_message_ratio'] = \
            float(person['from_this_person_to_poi'])/ \
            float(person['from_messages'])
        if float(person['to_messages']) > 0:
            person['from_poi_message_ratio'] = \
            float(person['from_poi_to_this_person'])/ \
            float(person['to_messages'])
        person['total_poi_interaction'] = person['shared_receipt_with_poi'] + \
        person['from_this_person_to_poi'] + person['from_poi_to_this_person']
        person['total_active_poi_interaction'] = \
        person['from_this_person_to_poi'] + person['from_poi_to_this_person'] 
        person['from_poi_total_active_poi_ratio'] = 0
        person['to_poi_total_active_poi_ratio'] = 0
        if person['total_active_poi_interaction'] > 0:
            person['from_poi_total_active_poi_ratio'] = \
            float(person['from_poi_to_this_person'])/ \
            float(person['total_active_poi_interaction'])
            person['to_poi_total_active_poi_ratio'] = \
            float(person['from_this_person_to_poi'])/ \
            float(person['total_active_poi_interaction'])
        #financial features        
        person['total_compensation'] = person['total_payments'] + \
                                   person['total_stock_value']
    '''#print new poi ratios   
    for key, person in data_dict.iteritems():
        print key, ": ", person['to_poi_message_ratio']
        print key, ": ", person['from_poi_message_ratio']
    '''    
    features_list.extend(['to_poi_message_ratio',
                          'from_poi_message_ratio',
                          'total_poi_interaction',
                          'total_active_poi_interaction',
                          'from_poi_total_active_poi_ratio',
                          'to_poi_total_active_poi_ratio',
                          'total_compensation'])
    return data_dict, features_list

def select_features(features, labels, features_list, k):
    # feature selection selectkbest
    selector = SelectKBest(f_classif, k=k).fit(features, labels)
    scores = selector.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    print "\nSelectKBest features:"
    kbest_features_list = dict(sorted_pairs[:k]).keys()
    
    print kbest_features_list, "\n"
    kbest_features_list.insert(0, 'poi')
    return kbest_features_list

def scale_features(features):
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    return features





### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

def create_clf_list():
    clf_list = []
    
    clf_tree = DecisionTreeClassifier()
    params_tree = { "min_samples_split":[2, 5, 10, 20],
                    "criterion": ('gini', 'entropy')
                    }
    clf_list.append( (clf_tree, params_tree) )

    #
    clf_linearsvm = LinearSVC()
    params_linearsvm = {"C": [0.5, 1, 5, 10, 100, 10**10],
                        "tol":[10**-1, 10**-10],
                        "class_weight":['balanced']

                        }
    clf_list.append( (clf_linearsvm, params_linearsvm) )

    #
    clf_adaboost = AdaBoostClassifier()
    params_adaboost = { "n_estimators":[20, 25, 50, 100]
                        }
    clf_list.append( (clf_adaboost, params_adaboost) )

    #
    clf_random_tree = RandomForestClassifier()
    params_random_tree = {  "n_estimators":[2, 3, 5],
                            "criterion": ('gini', 'entropy')
                            }
    clf_list.append( (clf_random_tree, params_random_tree) )

    #
    clf_knn = KNeighborsClassifier()
    params_knn = {"n_neighbors":[2, 5], "p":[2,3]}
    clf_list.append( (clf_knn, params_knn) )
    
    #
    clf_log = LogisticRegression()
    params_log = {  "C":[0.05, 0.5, 1, 10, 10**2,10**5,10**10],
                    "penalty":['l1','l2'],
                    "random_state": [42],                    
                    "tol":[10**-1, 10**-5, 10**-10],
                    "class_weight":['balanced']
                    }
    clf_list.append( (clf_log, params_log) )


    return clf_list
def create_pipeline(clf_list):

    pca = PCA()
    params_pca = {"pca__n_components":[2, 3, 4, 5, 10, 15, 20], 
                  "pca__whiten": [False]}

    for i in range(len(clf_list)):

        name = "clf_" + str(i)
        clf, params = clf_list[i]

        # For GridSearch to work with pipeline, the params have to have
        # double underscores between specific classifier and its parameter.
        new_params = {}
        for key, value in params.iteritems():
            new_params[name + "__" + key] = value

        new_params.update(params_pca)
        clf_list[i] = (Pipeline([("pca", pca), (name, clf)]), new_params)

    return clf_list
    
def run_clf(clf_list, features, labels, cv, metric='f1', v=0):

    n_iter_search = 20
    best_estimators = []
    for clf, params in clf_list:
        #clf = GridSearchCV(clf, params, cv=cv, n_jobs=-1, scoring=metric, verbose=v)
        print "Running RandomizedSearchCV"   
        clf = RandomizedSearchCV(clf, params, 
                                 n_iter=n_iter_search, cv=cv, 
                                 n_jobs=-1, scoring=metric, verbose=v)        
        clf = clf.fit(features, labels)
        best_estimators.append(clf.best_estimator_)

    return best_estimators

def evaluate_clf(clf, features_test, labels_test):


    labels_pred = clf.predict(features_test)

    f1 = f1_score(labels_test, labels_pred)
    recall = recall_score(labels_test, labels_pred)
    precision = precision_score(labels_test, labels_pred)
    return f1, recall, precision    


if __name__=="__main__":
    
    #Usage pass in # of iterations then True or False to use PCA
   
    
    #print len(sys.argv)      
    '''
    if len(sys.argv) != 3:  # command-line args is kept in sys.argv
        print len(sys.argv)        
        print('Usage: pass in # of iterations then True or False')
        sys.exit(1)  
    '''
    
    # data cleaning    
    data_dict = remove_outliers(data_dict)
    data_dict = correct_records(data_dict)    
    
    # exploratory data analysis & fillna's
    eda(data_dict, features_list)
    
    # add email and financial features
    data_dict, features_list = create_new_features(data_dict, features_list)
    
    # scale the features
    #features = scale_features(features)
    
    #use selectkbest to select top 10 features
    #features_list = select_features(features, labels, features_list, 10)
    
    
    my_dataset = data_dict
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    
    
    
    
    '''
    # Find the best classifier: StratifiedShuffleSplit for 1000 folds cv splits     
    cv = StratifiedShuffleSplit(labels, n_iter=1000, test_size=0.1)
            
    # generate list of classifiers and params
    clf_list = create_clf_list()
    
    evaluation_matrix = [[] for n in range(6)] # number of classifers
    
    print "Using Pipeline (PCA)"        
    clf_list = create_pipeline(clf_list)
    best_clf = run_clf(clf_list, features, labels, cv, 'f1', 3)
    
    for i, clf in enumerate(best_clf):
        scores = evaluate_clf(clf, features, labels)
        evaluation_matrix[i].append(scores)
    summary_list = {}
    for i, col in enumerate(evaluation_matrix):   
        summary_list[best_clf[i]] = (sum(asarray(col)))
    
    ordered_list = sorted(summary_list.keys() ,
                            key = lambda k: summary_list[k][0], reverse=True)
    
    clf = ordered_list[0]
    scores = summary_list[clf]
    print "Best classifier is ", clf
    print "With scores of f1, recall, precision: ", scores
    '''
    
    
    
    '''
    # Best classifier
    Pipeline(steps=[('pca', PCA(copy=True, n_components=20, whiten=False)), 
    ('clf_0', LogisticRegression(C=0.5, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l1', random_state=42,
          solver='liblinear', tol=1e-10, verbose=0, warm_start=False))])
    Accuracy: 0.80633       Precision: 0.36342      Recall: 0.60200 
    F1: 0.45323     F2: 0.53213
    Total predictions: 15000
    True positives: 1204
    False positives: 2109
    False negatives:  796
    True negatives: 10891
    '''
    # best performing classifier found by randomizedsearchcv
    clf_logistic = LogisticRegression(  C=.5,
                                        penalty='l1',
                                        random_state=42,
                                        tol=1e-10,
                                        class_weight='balanced')
    
    pca = PCA(n_components=20, whiten=False)
    
    clf = Pipeline(steps=[("pca", pca), ("logistic", clf_logistic)])
    


    test_classifier(clf, my_dataset, features_list)
    
    dump_classifier_and_data(clf, my_dataset, features_list)