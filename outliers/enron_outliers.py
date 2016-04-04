#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop("TOTAL", 0)
data = featureFormat(data_dict, features)

# List of all keys of the data_dict for  salary value > 1 million and bonus > 5 million dollars
key_list = [k for k in data_dict.keys() if data_dict[k]["salary"] != 'NaN' \
 and data_dict[k]["salary"] > 1000000]# and data_dict[k]["bonus"] > 5000000]

# Print the key values to find the outliers
for k in key_list:
    print k, " Salary: ", data_dict[k]["salary"], \
    " Bonus: ", data_dict[k]["bonus"], "POI: ", data_dict[k]["poi"]


for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
