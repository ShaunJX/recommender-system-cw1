import math
import sys
import os
import csv
import numpy as np
from operator import itemgetter

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

def get_similarity(user1: User, user2: User, common_item_threshold: int = 1):
    """
    This function returns the Pearson coefficient similarity between the given two users
    :param user1:
    :param user2:
    :param common_item_threshold: minimum number of common items (returns 0 if number of common items is lower)
    :return similarity (float):
    """
    user1_avg = user1.get_average_rating()
    user2_avg = user2.get_average_rating()

    common_items = user1.items.keys() & user2.items.keys()

    if len(common_items) < common_item_threshold:
        return 0

    sum_user1_x_user2 = 0
    sum_user1 = 0
    sum_user2 = 0

    for item_id in common_items:
        sum_user1_x_user2 += (user1.items[item_id].rating - user1_avg) * (user2.items[item_id].rating - user2_avg)
        sum_user1 += (user1.items[item_id].rating - user1_avg) ** 2
        sum_user2 += (user2.items[item_id].rating - user2_avg) ** 2

    return sum_user1_x_user2 / (math.sqrt(sum_user1) * math.sqrt(sum_user2))

def get_users_from_array(arr: np.ndarray):
    """
    This function returns a user_id to user dictionary given an array
    :param arr:
    :return users (dict):
    """
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

def get_neighbours(target_user: User, target_item_id, users: dict, maximum_neighbours: int = 10, common_item_threshold: int = 1, similarity_threshold: float = 0.7):
    neighbours = dict()
    for user_id in users:
        if user_id == target_user.user_id:
            continue
        if target_item_id not in users[user_id].items:
            continue
        similarity_with_target_user = get_similarity(target_user, users[user_id], common_item_threshold)
        if similarity_with_target_user > similarity_threshold:
            neighbours[user_id] = similarity_with_target_user

    if len(neighbours) <= 0 and similarity_threshold == 0.0:
        print("No neighbours found")
        return dict()

    if len(neighbours) <= 0:
        print("Similarity threshold too high")
        return dict()

    sorted_neighbours = dict(sorted(neighbours.items(), key=lambda item: item[1], reverse=True))
    target_neighbours = dict()

    while len(target_neighbours) < maximum_neighbours and len(sorted_neighbours) > 0:
        neighbour = list(sorted_neighbours.keys())[0]
        target_neighbours[neighbour] = sorted_neighbours.pop(neighbour)

    return target_neighbours

def predict_rating(target_user: User, target_item_id, users: dict, maximum_neighbours: int = 10, common_item_threshold: int = 1, similarity_threshold: float = 0.7):
    """
    This function predicts the rating that the target user will give the target item
    :param target_user:
    :param target_item_id:
    :param users:
    :param maximum_neighbours: maximum number of similar users to consider
    :param common_item_threshold: minimum number of common items
    :param similarity_threshold: minimum similarity rating to consider in prediction
    :return:
    """
    target_neighbours = get_neighbours(target_user, target_item_id, users, maximum_neighbours, common_item_threshold, similarity_threshold)

    if len(target_neighbours) == 0:
        print("Similarity threshold too high")
        target_neighbours = get_neighbours(target_user, target_item_id, users, maximum_neighbours, common_item_threshold, 0.0)

    if len(target_neighbours) == 0:
        print("No neighbours found, using target user's average rating")
        return target_user.average_rating

    similarity_sum = 0
    similarity_item_product_sum = 0

    for target_neighbour_id in target_neighbours:
        similarity = target_neighbours[target_neighbour_id]
        similarity_sum += similarity

        target_neighbour = users[target_neighbour_id]
        similarity_item_product_sum += similarity * (target_neighbour.items[target_item_id].rating - target_neighbour.average_rating)

    if similarity_sum == 0:
        print(f"Neighbours: {target_neighbours}")
        return target_user.average_rating

    return target_user.average_rating + (similarity_item_product_sum / similarity_sum)

def get_mae(train: np.ndarray , test: np.ndarray, maximum_neighbours: int = 10, common_item_threshold: int = 1, similarity_threshold: float = 0.7) -> float:
    users = get_users_from_array(train)

    absolute_error_sum = 0
    counter = 0

    for (user_id, item_id, rating, timestamp) in test:
        soft_class = predict_rating(users[user_id], item_id, users, maximum_neighbours, common_item_threshold, similarity_threshold)
        absolute_error_sum += abs(soft_class - rating)
        counter += 1

    return absolute_error_sum / counter

def cross_validation(arr: np.ndarray, nbins: int = 10, seed: int = 4):
    rng = np.random.default_rng(seed)
    rng.shuffle(arr)

    bins = dict()
    counter = 0
    for elem in arr:
        if counter not in bins:
            bins[counter] = []
        bins[counter].append(elem)
        counter += 1
        if counter >= nbins:
            counter = 0

    mean_absolute_errors = np.zeros(nbins)

    for i in range(nbins):
        test = bins[i]
        train = []
        for j in range(nbins):
            if i == j:
                continue
            if train == []:
                train = bins[j]
            else:
                train = train + bins[j]

        mean_absolute_errors[i] = get_mae(np.array(train), np.array(test))

    return np.average(mean_absolute_errors)




if __name__ == "__main__":
    train_arr = np.genfromtxt('train_100K.csv', delimiter=',')
    # user_ratings = get_user_ratings(1, train_arr)
    #
    # print(f'average rating: {get_average_rating(user_ratings)}')
    # array = np.array([[1,1,8,1],[1,4,2,1],[1,5,7,1],
    #                [2,1,2,1],[2,3,5,1],[2,4,7,1],[2,5,5,1],
    #                [3,1,5,1],[3,2,4,1],[3,3,7,1],[3,4,4,1],[3,5,7,1],
    #                [4,1,7,1],[4,2,1,1],[4,3,7,1],[4,4,3,1],[4,5,8,1],
    #                [5,1,1,1],[5,2,7,1],[5,3,4,1],[5,4,6,1],[5,5,5,1],
    #                [6,1,8,1],[6,2,3,1],[6,3,8,1],[6,4,3,1],[6,5,7,1]])
    print(cross_validation(train_arr, nbins=10))