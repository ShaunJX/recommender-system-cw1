import math
import sys
import os
import csv
import numpy as np

#declare your own function(s)
def func1():
    print("Test")

if __name__ == "__main__":
    func1()

def get_user_ratings(user, arr):
    """
    This function returns a dictionary of item id to rating of the given user
    :param user: id of the user
    :param arr:
    :return:
    """
    user_ratings = dict()
    for elem in arr:
        # elem[0]: user id
        # elem[1]: item id
        # elem[2]: rating
        if elem[0] == user:
            user_ratings[elem[1]] = elem[2]
    return user_ratings

def get_average_rating(user_ratings:dict):
    """
    This function returns the average rating of the given the item to rating dictionary
    :param user_ratings: item id to rating dictionary
    :return: average rating
    """
    counter = 0
    _sum = 0
    for item in user_ratings.keys():
        _sum += user_ratings[item]
        counter += 1
    return _sum / counter