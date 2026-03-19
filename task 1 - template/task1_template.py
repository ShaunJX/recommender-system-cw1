import math
import sys
import os
import csv
import numpy as np

#declare your own function(s)
def func1():
    print("Test")



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

def get_similarity(user1, user2, arr):
    """
    This function returns the Pearson similarity between the given two users
    :param user1:
    :param user2:
    :param arr:
    :return:
    """

    def get_user_ratings(_user1, _user2, _arr):
        """
        This function returns a dictionary of item id to rating of the given users
        :param _user1:
        :param _user2:
        :param _arr:
        :return:
        """
        user1_ratings = dict()
        user2_ratings = dict()
        for elem in _arr:
            # elem[0]: user id
            # elem[1]: item id
            # elem[2]: rating
            if elem[0] == _user1:
                user1_ratings[elem[1]] = elem[2]
            elif elem[0] == _user2:
                user2_ratings[elem[1]] = elem[2]
        return user1_ratings, user2_ratings

    user1_ratings, user2_ratings = get_user_ratings(user1, user2, arr)
    user1_avg = get_average_rating(user1_ratings)
    user2_avg = get_average_rating(user2_ratings)

    common_items = user1_ratings.keys() & user2_ratings.keys()

    sum_user1_x_user2 = 0
    sum_user1 = 0
    sum_user2 = 0

    for item in common_items:
        sum_user1_x_user2 += (user1_ratings[item] - user1_avg) * (user2_ratings[item] - user2_avg)
        sum_user1 += (user1_ratings[item] - user1_avg) ** 2
        sum_user2 += (user2_ratings[item] - user2_avg) ** 2

    return sum_user1_x_user2 / (math.sqrt(sum_user1) * math.sqrt(sum_user2))


if __name__ == "__main__":
    # train_arr = np.genfromtxt('train_100K.csv', delimiter=',')
    # user_ratings = get_user_ratings(1, train_arr)
    #
    # print(f'average rating: {get_average_rating(user_ratings)}')
    array = np.array([[1,1,8],[1,4,2],[1,5,7],
                    [2,1,2],[2,3,5],[2,4,7],[2,5,5],
                    [3,1,5],[3,2,4],[3,3,7],[3,4,4],[3,5,7],
                    [4,1,7],[4,2,1],[4,3,7],[4,4,3],[4,5,8],
                    [5,1,1],[5,2,7],[5,3,4],[5,4,6],[5,5,5],
                    [6,1,8],[6,2,3],[6,3,8],[6,4,3],[6,5,7]])
    print(get_similarity(1,4, array))