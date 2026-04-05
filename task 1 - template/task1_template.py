import math
import sys
import os
import csv
import numpy as np
import time

ratings = dict()  # dictionary that takes two keys (user_id, item_id) and returns two values (rating, timestamp)
user_similarity_matrix = dict() # dictionary that takes two keys (user_id1, user_id2) and returns the similarity between them
item_similarity_matrix = dict() # dictionary that takes two keys (item_id1, item_id2) and returns the similarity between them

class Item:
    def __init__(self, item_id):
        self.item_id = item_id
        self.rated_users = set() # set of user_id who rated this item

class User:
    def __init__(self, user_id):
        self.user_id = user_id
        self.items_rated = set() # set of item_id that user rated
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
        for item_id in self.items_rated:
            _sum += ratings[(self.user_id, item_id)][0]

        self.average_rating = _sum / len(self.items_rated)
        return self.average_rating

def get_user_similarity(user1: User, user2: User, common_item_threshold: int = 1):
    """
    This function returns the Pearson coefficient similarity between the given two users
    :param user1:
    :param user2:
    :param common_item_threshold: minimum number of common items (returns 0 if number of common items is lower)
    :return similarity (float):
    """
    if (user1.user_id, user2.user_id) in user_similarity_matrix:
        return user_similarity_matrix[(user1.user_id, user2.user_id)]

    user1_avg = user1.get_average_rating()
    user2_avg = user2.get_average_rating()

    common_items = user1.items_rated & user2.items_rated

    if len(common_items) < common_item_threshold:
        return 0

    sum_user1_x_user2 = 0
    sum_user1 = 0
    sum_user2 = 0

    for item_id in common_items:
        user1_rating = ratings[(user1.user_id, item_id)][0]
        user2_rating = ratings[(user2.user_id, item_id)][0]
        sum_user1_x_user2 += (user1_rating - user1_avg) * (user2_rating - user2_avg)
        sum_user1 += (user1_rating - user1_avg) ** 2
        sum_user2 += (user2_rating - user2_avg) ** 2

    result = sum_user1_x_user2 / (math.sqrt(sum_user1) * math.sqrt(sum_user2))
    user_similarity_matrix[(user1.user_id, user2.user_id)] = result
    user_similarity_matrix[(user2.user_id, user1.user_id)] = result
    return result

def read_array(arr: np.ndarray):
    """
    This function returns a user_id to user dictionary and an item_id to item dictionary given an array
    :param arr:
    :return users (dict):
    :return items (dict):
    """
    users = dict() # dictionary that takes a key (user_id) and returns a value (User)
    items = dict() # dictionary that takes a key (item_id) and returns a value (Item)

    for elem in arr:
        user_id = int(elem[0])
        item_id = int(elem[1])
        rating = np.float16(elem[2])
        timestamp = int(elem[3])

        if user_id not in users:
            users[user_id] = User(user_id)

        if item_id not in items:
            items[item_id] = Item(item_id)

        users[user_id].items_rated.append(item_id)
        items[item_id].rated_users.append(user_id)
        ratings[(user_id, item_id)] = (rating, timestamp)

    return users, items

def get_neighbours(target_user: User, target_item_id, users: dict, maximum_neighbours: int = 10, common_item_threshold: int = 1, similarity_threshold: float = 0.7):
    neighbours = dict()
    for user_id in users:
        if user_id == target_user.user_id:
            continue
        if target_item_id not in users[user_id].items:
            continue
        similarity_with_target_user = get_user_similarity(target_user, users[user_id], common_item_threshold)
        if similarity_with_target_user > similarity_threshold:
            neighbours[user_id] = similarity_with_target_user

    if len(neighbours) <= 0 and similarity_threshold == 0.0:
        return dict()

    if len(neighbours) <= 0:
        return dict()

    sorted_neighbours = dict(sorted(neighbours.items(), key=lambda item: item[1], reverse=True))
    target_neighbours = dict()

    while len(target_neighbours) < maximum_neighbours and len(sorted_neighbours) > 0:
        neighbour = list(sorted_neighbours.keys())[0]
        target_neighbours[neighbour] = sorted_neighbours.pop(neighbour)

    return target_neighbours

def get_item_neighbours():
    pass

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
        target_neighbours = get_neighbours(target_user, target_item_id, users, maximum_neighbours, common_item_threshold, 0.0)

    if len(target_neighbours) == 0:
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

    soft_class = target_user.average_rating + (similarity_item_product_sum / similarity_sum)
    hard_class = round(soft_class / 0.5) * 0.5

    return hard_class

def get_mae(train: np.ndarray , test: np.ndarray, maximum_neighbours: int = 10, common_item_threshold: int = 1, similarity_threshold: float = 0.7) -> float:
    users, items = read_array(train)

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
        print(f"Starting fold {i}")
        start_time = time.time()
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
        end_time = time.time()
        print(f"Fold {i} took {end_time - start_time} seconds")
        print(f"Fold {i} result: {mean_absolute_errors}")

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