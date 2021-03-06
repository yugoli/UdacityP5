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
from sklearn.svm import LinearSVC
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
    print "\nExploring Dataset:"
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
    
    top = []
    for key in data_dict:
        #val = data_dict[key]['bonus']
        val = data_dict[key]['salary']
        if val == 'NaN':
            continue;    
        top.append((key,int(val)))
    #print top 3 bonuses    
    print (sorted(top,key=lambda x:x[1],reverse=True)[:3]), "\n"

def remove_outliers(data_dict):
    ### Task 2: Remove outliers
    print "Removed outliers."
    outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
    for outlier in outliers:
        data_dict.pop(outlier, 0)
    return data_dict
def correct_records(data_dict):
    #Fix two records
    print "Corrected 2 records."
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
    '''
    #
    clf_nb = GaussianNB()
    params_nb = {}
    clf_list.append( (clf_nb, params_nb) )
    #
    clf_linearsvm = LinearSVC()
    params_linearsvm = {"C": [0.5, 1, 5, 10, 100, 10**10],
                        "tol":[10**-1, 10**-10],
                        "class_weight":['balanced']

                        }
    clf_list.append( (clf_linearsvm, params_linearsvm) )

    #
    clf_knn = KNeighborsClassifier()
    params_knn = {"n_neighbors":[2, 5], "p":[2,3]}
    clf_list.append( (clf_knn, params_knn) )
    '''                                                                                                        
    #
    clf_log = LogisticRegression()
    params_log = {  "C":[0.05, 0.5, 1, 10],
                    "penalty":['l1','l2'],
                    "random_state": [42],                    
                    "tol":[10**-1, 10**-5, 10**-10],
                    "class_weight":['balanced']
                    }
    clf_list.append( (clf_log, params_log) )


    return clf_list
def create_pipeline(clf_list):

    pca = PCA()
    scaler = MinMaxScaler()
    selector = SelectKBest(f_classif)
    params_update = {"kbest__k" : range(5, 26),
                     "pca__n_components":[1, 2, 3, 4, 5], 
                     "pca__whiten": [False]}

    for i in range(len(clf_list)):

        name = "clf_" + str(i)
        clf, params = clf_list[i]

        # For GridSearch to work with pipeline, the params have to have
        # double underscores between specific classifier and its parameter.
        new_params = {}
        for key, value in params.iteritems():
            new_params[name + "__" + key] = value

        new_params.update(params_update)
        clf_list[i] = (Pipeline([('kbest', selector),
                        ("scaler",scaler),
                        ("pca", pca), 
                        (name, clf)]), new_params)

    return clf_list
    
def run_clf(clf_list, features, labels, cv, metric='f1', v=0, random=False):

    n_iter_search = 20
    best_estimators = []
    for clf, params in clf_list:
        if random:        
            print "\nRunning RandomizedSearchCV on \n", clf
            print "\n", params 
            clf = RandomizedSearchCV(clf, params, 
                                     n_iter=n_iter_search, cv=cv, 
                                     n_jobs=-1, scoring=metric, verbose=v)        
        else:
            print "\nRunning GridSearchCV on \n", clf
            print "\n", params           
            clf = GridSearchCV(clf, params, cv=cv, n_jobs=-1, 
                               scoring=metric, verbose=v)
        clf = clf.fit(features, labels)
        print "Best parameters: \n", clf.best_params_        
        best_estimators.append(clf.best_estimator_)

    return best_estimators

def evaluate_clf(clf, features, labels):


    labels_pred = clf.predict(features)
    f1 = f1_score(labels, labels_pred)
    recall = recall_score(labels, labels_pred)
    precision = precision_score(labels, labels_pred)
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
    
    # add email and financial features to list
    data_dict, features_list = create_new_features(data_dict, features_list)
    
    # create features from dataset
    my_dataset = data_dict
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    
    # scale the features - implemented in pipeline
    #features = scale_features(features)
    
    #use selectkbest to select top features  - implemented in pipeline
    #features_list = select_features(features, labels, features_list, 20)
    
    
    # Find the best classifier: StratifiedShuffleSplit for 100 folds cv splits     
    cv = StratifiedShuffleSplit(labels, n_iter=100, test_size=0.1, \
                                random_state=42)
    

    # generate list of classifiers and params
    clf_list = create_clf_list()
    
    evaluation_matrix = [[] for n in range(4)] # number of classifers
    
    print "Using Pipeline"        
    clf_list = create_pipeline(clf_list)
    #best_clf = run_clf(clf_list, features, labels, cv, 'f1', 3, True)
    best_clf = run_clf(clf_list, features, labels, cv, 'f1', 3)
    
    for i, clf in enumerate(best_clf):
        scores = evaluate_clf(clf, features, labels)
        print clf, "\n", scores        
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
    
    
    
    ''' Results
    {'clf_3__C': 10, 'clf_3__class_weight': 'balanced', 'pca__n_components': 2, 'pca__whiten': False, 'clf_3__tol': 1e-05, 'kbest__k': 13, 'clf_3__penalty': 'l2', 'clf_3__random_state': 42}
Pipeline(steps=[('kbest', SelectKBest(k=13, score_func=<function f_classif at 0x000000000994C828>)), ('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('pca', PCA(copy=True, n_components=4, whiten=False)), ('clf_0', GaussianNB())]) 
(0.35294117647058826, 0.33333333333333331, 0.375)
Pipeline(steps=[('kbest', SelectKBest(k=7, score_func=<function f_classif at 0x000000000994C828>)), ('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('pca', PCA(copy=True, n_components=3, whiten=False)), ('clf_1', LinearSVC(C=5, class_weight='balanced', dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=1e-10,
     verbose=0))]) 
(0.51851851851851849, 0.77777777777777779, 0.3888888888888889)
Pipeline(steps=[('kbest', SelectKBest(k=24, score_func=<function f_classif at 0x000000000994C828>)), ('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('pca', PCA(copy=True, n_components=1, whiten=False)), ('clf_2', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=3,
           weights='uniform'))]) 
(0.34782608695652178, 0.22222222222222221, 0.80000000000000004)
Pipeline(steps=[('kbest', SelectKBest(k=13, score_func=<function f_classif at 0x000000000994C828>)), ('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('pca', PCA(copy=True, n_components=2, whiten=False)), ('clf_3', LogisticRegression(C=10, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=42,
          solver='liblinear', tol=1e-05, verbose=0, warm_start=False))]) 
(0.44444444444444442, 0.66666666666666663, 0.33333333333333331)

    # Best classifier found by Randomizedsearchcv 
    Best classifier is  Pipeline(steps=[('kbest', 
    SelectKBest(k=7, score_func=<function f_classif at 0x000000000994C828>)), 
    ('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), 
    ('pca', PCA(copy=True, n_components=3, whiten=False)), 
    ('clf_1', LinearSVC(C=5, class_weight='balanced', dual=True, 
    fit_intercept=True, intercept_scaling=1, loss='squared_hinge', 
    max_iter=1000, multi_class='ovr', penalty='l2', random_state=None, 
    tol=1e-10, verbose=0))])
    With scores of f1, recall, precision: [ 0.51851852  0.77777778  0.38888889]
    tester.py results
    Accuracy: 0.77080    Precision: 0.32913   Recall: 0.69250 
    F1: 0.44620     F2: 0.56725
    Total predictions: 15000   True positives: 1385 False positives: 2823   
    False negatives:  615   True negatives: 10177
    '''
    '''
    # Best classifier found by GridSearchCV
    
    {'clf_3__C': 100, 'clf_3__class_weight': 'balanced', 'pca__n_components': 3, 'pca__whiten': False, 'clf_3__tol': 1e-05, 'kbest__k': 10, 'clf_3__penalty': 'l1', 'clf_3__random_state': 42}
Pipeline(steps=[('kbest', SelectKBest(k=10, score_func=<function f_classif at 0x000000000994C828>)), ('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('pca', PCA(copy=True, n_components=5, whiten=False)), ('clf_0', GaussianNB())]) 
(0.375, 0.33333333333333331, 0.42857142857142855)
Pipeline(steps=[('kbest', SelectKBest(k=8, score_func=<function f_classif at 0x000000000994C828>)), ('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('pca', PCA(copy=True, n_components=3, whiten=False)), ('clf_1', LinearSVC(C=10, class_weight='balanced', dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.1,
     verbose=0))]) 
(0.51724137931034475, 0.83333333333333337, 0.375)
Pipeline(steps=[('kbest', SelectKBest(k=6, score_func=<function f_classif at 0x000000000994C828>)), ('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('pca', PCA(copy=True, n_components=1, whiten=False)), ('clf_2', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform'))]) 
(0.34482758620689657, 0.27777777777777779, 0.45454545454545453)
Pipeline(steps=[('kbest', SelectKBest(k=10, score_func=<function f_classif at 0x000000000994C828>)), ('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('pca', PCA(copy=True, n_components=3, whiten=False)), ('clf_3', LogisticRegression(C=100, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l1', random_state=42,
          solver='liblinear', tol=1e-05, verbose=0, warm_start=False))]) 
(0.5, 0.83333333333333337, 0.35714285714285715)


    Best classifier is  Pipeline(steps=[('kbest', SelectKBest(k=8, 
    score_func=<function f_classif at 0x000000000994C828>)), 
    ('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), 
    ('pca', PCA(copy=True, n_components=3, whiten=False)), 
    ('clf_1', LinearSVC(C=10, class_weight='balanced', dual=True, 
    fit_intercept=True, intercept_scaling=1, loss='squared_hinge', 
    max_iter=1000, multi_class='ovr', penalty='l2', random_state=None, tol=0.1,
    verbose=0))])
    With scores of f1, recall, precision: [ 0.51724138  0.83333333  0.375     ]
    tester.py results
    Accuracy: 0.76507    Precision: 0.32483   Recall: 0.70650 
    F1: 0.44504     F2: 0.57206
    Total predictions: 15000        True positives: 1413    
    False positives: 2937   False negatives:  587   True negatives: 10063
    '''
    
    print "\n******************"

    test_classifier(clf, my_dataset, features_list)
    
    dump_classifier_and_data(clf, my_dataset, features_list)