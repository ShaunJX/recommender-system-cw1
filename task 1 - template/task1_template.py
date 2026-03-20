import math
import sys
import os
import csv
import numpy as np

class Item:
    def __init__(self, item_id, rating, timestamp):
        self.item_id = item_id
        self.rating = rating
        self.timestamp = timestamp

class User:
    def __init__(self, user_id):
        self.user_id = user_id
        self.items = dict() # item_id to item dictionary
        self.average_rating = None

    def get_average_rating(self):
        """
        This function returns the average rating given by the user.
        Once the average rating is calculated, the result is saved and this function will then return the saved result.
        Therefore, only run this function once all item ratings have been added, otherwise the average rating would be incorrect.
        :return average rating (float):
        """
        if self.average_rating is not None:
            return self.average_rating

        _sum = 0
        for item_id in self.items:
            _sum += self.items[item_id].rating

        self.average_rating = _sum / len(self.items)
        return self.average_rating

def get_similarity(user1: User, user2: User):
    """
    This function returns the Pearson coefficient similarity between the given two users
    :param user1:
    :param user2:
    :return similarity (float):
    """
    user1_avg = user1.get_average_rating()
    user2_avg = user2.get_average_rating()

    common_items = user1.items.keys() & user2.items.keys()

    sum_user1_x_user2 = 0
    sum_user1 = 0
    sum_user2 = 0

    for item_id in common_items:
        sum_user1_x_user2 += (user1.items[item_id] - user1_avg) * (user2.items[item_id] - user2_avg)
        sum_user1 += (user1.items[item_id] - user1_avg) ** 2
        sum_user2 += (user2.items[item_id] - user2_avg) ** 2

    return sum_user1_x_user2 / (math.sqrt(sum_user1) * math.sqrt(sum_user2))

def get_users_from_csv(file_path):
    arr = np.genfromtxt(file_path, delimiter=',')

    users = dict()

    for elem in arr:
        user_id = int(elem[0])
        item_id = int(elem[1])
        rating = np.float16(elem[2])
        timestamp = int(elem[3])

        if user_id not in users:
            users[user_id] = User(user_id)

        users[user_id].items[item_id] = Item(item_id, rating, timestamp)

    return users

def predict_rating(user: User, item_id, users: dict):
    pass

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