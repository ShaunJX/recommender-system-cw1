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
    global ratings
    global item_similarity_matrix
    global user_similarity_matrix
    ratings = dict()
    item_similarity_matrix = dict()
    user_similarity_matrix = dict()

    for elem in arr:
        user_id = int(elem[0])
        item_id = int(elem[1])
        rating = np.float16(elem[2])
        timestamp = int(elem[3])

        if user_id not in users:
            users[user_id] = User(user_id)

        if item_id not in items:
            items[item_id] = Item(item_id)

        users[user_id].items_rated.add(item_id)
        items[item_id].rated_users.add(user_id)
        ratings[(user_id, item_id)] = (rating, timestamp)

    return users, items

def get_neighbours(target_user: User, target_item_id, users: dict, maximum_neighbours: int = 10, common_item_threshold: int = 1, similarity_threshold: float = 0.7):
    neighbours = dict()
    for user_id in users:
        if user_id == target_user.user_id:
            continue
        if target_item_id not in users[user_id].items_rated:
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

def get_item_similarity(item1: Item, item2: Item, users: dict):
    if (item1.item_id, item2.item_id) in item_similarity_matrix:
        return item_similarity_matrix[(item1.item_id, item2.item_id)]

    common_users = item1.rated_users & item2.rated_users

    if len(common_users) <= 0:
        return 0

    sum_item1_x_item2 = 0
    sum_item1 = 0
    sum_item2 = 0
    for user_id in common_users:
        if user_id not in users:
            continue
        item1_rating = ratings[(user_id, item1.item_id)][0]
        item2_rating = ratings[(user_id, item2.item_id)][0]
        user_avg_rating = users[user_id].get_average_rating()
        sum_item1_x_item2 += (item1_rating - user_avg_rating) * (item2_rating - user_avg_rating)
        sum_item1 += (item1_rating - user_avg_rating) ** 2
        sum_item2 += (item2_rating - user_avg_rating) ** 2

    if sum_item1 == 0 or sum_item2 == 0:
        return 0

    item_similarity = sum_item1_x_item2 / (math.sqrt(sum_item1) * math.sqrt(sum_item2))
    item_similarity_matrix[(item1.item_id, item2.item_id)] = item_similarity
    item_similarity_matrix[(item2.item_id, item1.item_id)] = item_similarity
    return item_similarity

def get_item_neighbours(target_item: Item, target_user_id, items: dict, users: dict, maximum_neighbours: int = 10, similarity_threshold: float = 0.7):
    neighbours = dict()
    for item_id in items:
        if item_id == target_item.item_id:
            continue
        if target_user_id not in items[item_id].rated_users:
            continue
        similarity_with_target_item = get_item_similarity(target_item, items[item_id], users)
        if similarity_with_target_item > similarity_threshold:
            neighbours[item_id] = similarity_with_target_item

    if len(neighbours) <= 0:
        return dict()

    sorted_neighbours = dict(sorted(neighbours.items(), key=lambda item: item[1], reverse=True))
    target_neighbours = dict()

    while len(target_neighbours) < maximum_neighbours and len(sorted_neighbours) > 0:
        neighbour = list(sorted_neighbours.keys())[0]
        target_neighbours[neighbour] = sorted_neighbours.pop(neighbour)

    return target_neighbours

def get_item_based_prediction(target_item: Item, target_user_id, items: dict, users: dict, maximum_neighbours: int = 10, similarity_threshold: float = 0.7):
    target_neighbours = get_item_neighbours(target_item, target_user_id, items, users, maximum_neighbours, similarity_threshold)

    if len(target_neighbours) == 0:
        target_neighbours = get_item_neighbours(target_item, target_user_id, items, users, maximum_neighbours, 0.0)

    if len(target_neighbours) == 0:
        return users[target_user_id].get_average_rating()

    similarity_sum = 0
    similarity_rating_product_sum = 0

    for target_neighbour_id in target_neighbours:
        similarity = target_neighbours[target_neighbour_id]
        similarity_sum += similarity

        rating = ratings[(target_user_id, target_neighbour_id)][0]
        similarity_rating_product_sum += similarity * rating

    if similarity_sum == 0:
        print(f"Neighbours: {target_neighbours}")
        return users[target_user_id].get_average_rating()

    soft_class = similarity_rating_product_sum / similarity_sum
    hard_class = round(soft_class / 0.5) * 0.5

    return hard_class

def predict_rating(target_user: User, target_item_id, items: dict, users: dict, maximum_neighbours: int = 10, common_item_threshold: int = 1, similarity_threshold: float = 0.7):
    """
    This function predicts the rating that the target user will give the target item
    :param target_user:
    :param target_item_id:
    :param items:
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
        if target_item_id in items:
            return get_item_based_prediction(items[target_item_id], target_user.user_id, items, users, maximum_neighbours, similarity_threshold)
        else:
            return target_user.get_average_rating()
        #return target_user.get_average_rating()

    similarity_sum = 0
    similarity_item_product_sum = 0

    for target_neighbour_id in target_neighbours:
        similarity = target_neighbours[target_neighbour_id]
        similarity_sum += similarity

        target_neighbour = users[target_neighbour_id]
        similarity_item_product_sum += similarity * (ratings[(target_neighbour_id, target_item_id)][0] - target_neighbour.get_average_rating())

    if similarity_sum == 0:
        print(f"Neighbours: {target_neighbours}")
        return target_user.average_rating

    soft_class = target_user.get_average_rating() + (similarity_item_product_sum / similarity_sum)
    hard_class = round(soft_class / 0.5) * 0.5

    return hard_class

def get_mae(train: np.ndarray , test: np.ndarray, maximum_neighbours: int = 10, similarity_threshold: float = 0.7) -> float:
    users, items = read_array(train)

    absolute_error_sum = 0
    counter = 0

    for (user_id, item_id, rating, timestamp) in test:
        prediction = predict_rating(users[user_id], item_id, items, users, maximum_neighbours, 1, similarity_threshold)
        absolute_error_sum += abs(prediction - rating)
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

def get_prediction(train: np.ndarray, test: np.ndarray, maximum_neighbours: int = 10, common_item_threshold: int = 1, similarity_threshold: float = 0.7):
    users, items = read_array(train)

    predicted_arr = np.zeros(len(test))

    for i in range(len(test)):
        user_id = test[i][0]
        item_id = test[i][1]
        timestamp = test[i][2]

        prediction = predict_rating(users[user_id], item_id, items, users, maximum_neighbours, common_item_threshold, similarity_threshold)
        predicted_arr[i] = np.array([int(user_id), int(item_id), prediction, int(timestamp)])

    return predicted_arr

def create_csv_from_arr(arr: np.ndarray, fname):
    with open(fname, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(arr)

if __name__ == "__main__":
    train_arr = np.genfromtxt('train_100K.csv', delimiter=',')
    test_arr = np.genfromtxt('test_100K.csv', delimiter=',')

    # print(cross_validation(train_arr, nbins=5)) # Evaluation via cross validation

    result = get_prediction(train_arr, test_arr)
    create_csv_from_arr(result, 'result.csv')