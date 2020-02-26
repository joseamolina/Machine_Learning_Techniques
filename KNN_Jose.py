import numpy as np
import pandas as pd
from math import sqrt

class KnnCancerClassifier:

    # Initialize class variables
    def __init__(self, file_name):

        # Read from data set to a numpy array
        self.data_set_cancer = np.genfromtxt(file_name, delimiter=',')
        self.k = None
        self.entries_data = np.shape(self.data_set_cancer)[0]
        self.len_data = np.shape(self.data_set_cancer[0])[0]
        self.testing_data_set = None
        self.len_test_data = None
        self.applier_distance = None
        self.predictor_normal = True

    # This method switches the k value for normal knn depending on passing parameter.
    def _set_k(self, k_measure: "string") -> "None":
        switcher = {
            'static': 3,
            'squared': int(np.round(np.sqrt(self.len_data))),
            'n-fold': int(np.round(self.len_data / self.len_test_data)) + 1
        }

        self.k = switcher.get(k_measure)

    # This method switches the distance calculating parameter depending on passing parameter.
    def _set_distance(self, distance: "string") -> "None":
        switcher = {
            'euclidean': self._distance_euclidean,
            'manhattan': self._distance_manhattan,
            'chebyshev': self._distance_chebyshev,
            'canberra': self._distance_canberra,
            'braycurtis': self._distance_braycurtis
        }
        self.applier_distance = switcher.get(distance)

    # Calculates the euclidean distance between 2 vectors
    def _distance_euclidean(self, c: "Int", target) -> "Double":

        # Parametrize data
        target_new = list(map(lambda r: self._parametize_data(r[1], r[0]), enumerate(target[:self.len_data - 1])))
        entry_new = list(map(lambda r: self._parametize_data(r[1], r[0]), enumerate(self.data_set_cancer[c, :self.len_data - 1])))

        return np.sqrt(np.sum(np.square(np.subtract(target_new, entry_new))))

    # Calculates the manhattan distance between 2 vectors
    def _distance_manhattan(self, c: "Int", target) -> "Double":

        # Parametrize data
        target_new = list(map(lambda r: self._parametize_data(r[1], r[0]), enumerate(target[:self.len_data - 1])))
        entry_new = list(
            map(lambda r: self._parametize_data(r[1], r[0]), enumerate(self.data_set_cancer[c, :self.len_data - 1])))

        return np.sum(np.abs(np.subtract(target_new, entry_new)))

    # Calculates the chebyshev distance between 2 vectors
    def _distance_chebyshev(self, c: "Int", target) -> "Double":

        # Parametrize data
        target_new = list(map(lambda r: self._parametize_data(r[1], r[0]), enumerate(target[:self.len_data - 1])))
        entry_new = list(
            map(lambda r: self._parametize_data(r[1], r[0]), enumerate(self.data_set_cancer[c, :self.len_data - 1])))

        return np.max(np.abs(np.subtract(target_new, entry_new)))

    # Calculates the canberra distance between 2 vectors
    def _distance_canberra(self, c: "Int", target) -> "Double":
        return max(abs(self._parametize_data(target[i], i) - self._parametize_data(self.data_set_cancer[c, i], i)) / (abs(
         self._parametize_data(target[i], i)) + abs(self._parametize_data(self.data_set_cancer[c, i], i))) for i in
          range(self.len_data - 1))

    # Calculates the braycurtis distance between 2 vectors
    def _distance_braycurtis(self, c: "Int", target) -> "Double":
        return max(abs(self._parametize_data(target[i], i) - self._parametize_data(self.data_set_cancer[c, i], i)) / (sum(
            self._parametize_data(target[k], k) for k in range(self.len_data - 1)) + sum(self._parametize_data(
                self.data_set_cancer[c, y], y) for y in range(self.len_data - 1))) for i in range(self.len_data - 1))

    # This method says if a prediction of a target query is valid or not. Normal distance
    def _calculateDistances(self, target_ob):

        # Create a Series data structure containing indexes to be measured
        distances = pd.Series(range(self.entries_data))

        # For each content of the series, calculate the distance between the target query and the indexed point
        distances_computed = distances.apply(lambda x: self.applier_distance(x, target_ob))

        # To sort the indexes according to the values contained (distances between target and each entry)
        sorted_distances = np.argsort(distances_computed)

        # Get the n first entries and count them
        counts = np.bincount([self.data_set_cancer[i, 5] for i in sorted_distances[:self.k]])

        # Return T if the prediction is valid
        return counts.argmax() == target_ob[5]

    # This method says if a prediction of a target query is valid or not. Weighted distance
    def _predict_weighted(self, target_ob, n):

        # Create a Series data structure containing indexes to be measured
        distances = pd.Series(range(self.entries_data))

        # For each entry, create a Tuple (class, 1/distance); being distance the distance between the entry
        # and the target
        distances_computed = distances.apply(lambda x: (self.data_set_cancer[x, 5], 1/(self.applier_distance(x, target_ob))**n))
        # To sum for each class the frequency
        sum_0 = 0
        sum_1 = 0
        for i in distances_computed:
            if i[0] == 0:
                sum_0 += i[1]
            else:
                sum_1 += i[1]

        # To guess what class wins
        class_pred = 0
        if sum_1 >= sum_0:
            class_pred = 1

        # Return T if the prediction is valid
        return class_pred == target_ob[5]

    def test_accuracy(self, file, k_set, distance):

        # Read instances of testing data set
        self.testing_data_set = np.genfromtxt(file, delimiter=',')

        # Get references of testing
        self.len_test_data = np.shape(self.testing_data_set)[0]

        self._set_k(k_set)
        self._set_distance(distance)

        # Create set containing the distances calculated for each testing entry
        prediction_applied = pd.Series(range(self.len_test_data))
        list_bool = prediction_applied.apply(lambda x: self._calculateDistances(self.testing_data_set[x]))

        # Print results
        print("Accuracy of the model: {0}%".format((list_bool.value_counts().get(True) * 100) / self.len_test_data))
        print("Parameters: Distance: {0}, K format: {1}\n\n".format(distance, k_set))

    def test_accuracy_weighted(self, file, distance, n):

        self.testing_data_set = np.genfromtxt(file, delimiter=',')
        self.len_test_data = np.shape(self.testing_data_set)[0]

        self._set_distance(distance)

        prediction_applied = pd.Series(range(self.len_test_data))
        list_bool = prediction_applied.apply(lambda x: self._predict_weighted(self.testing_data_set[x], n))

        print("Accuracy of the model: {0}%".format((list_bool.value_counts().get(True) * 100) / self.len_test_data))
        print("Parameters: Distance: {0} and n= {1}\n\n".format(distance, n))

    # This method normalizes the data according to its parameters range
    @staticmethod
    def _parametize_data(obj, index):
        switcher = {
            0: (obj - 1) / (5 - 1),
            1: (obj - 1) / (120 - 1),
            2: (obj - 1) / (4 - 1),
            3: (obj - 1) / (5 - 1),
            4: (obj - 1) / (4 - 1)
        }
        return switcher.get(index)

# Initialize instance
dert = KnnCancerClassifier('cancer/trainingData2.csv')

# Battery of normal classification
for k in ['static', 'squared', 'n-fold']:
    for tech in ['euclidean', 'manhattan', 'chebyshev', 'canberra', 'braycurtis']:
        dert.test_accuracy('cancer/testData2.csv', k, tech)

# Battery of weighted classification
for i in range(1,4):
    for tech in ['euclidean', 'manhattan', 'chebyshev', 'canberra', 'braycurtis']:
        dert.test_accuracy_weighted('cancer/testData2.csv', tech, i)