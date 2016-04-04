#!/usr/bin/python

from operator import itemgetter

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    
    errors = list((net_worths- predictions)**2)
    
    cleaned_data = zip(ages, net_worths, errors)
        
    #cleaned_data = sorted(cleaned_data, key = lambda tup: tup[2])
    cleaned_data = sorted(cleaned_data, key=itemgetter(2) ) 
    # using itemgetter sorted by 3 element of tuple
    cleaned_data = cleaned_data[:81]
    
    print cleaned_data, len(cleaned_data)
    ### your code goes here
    
    
    return cleaned_data

