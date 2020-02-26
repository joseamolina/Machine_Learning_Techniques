import numpy as np
from math import sqrt
import pandas as pd

class KNN_regresion_Jose:

    # Initialize class variables
    def __init__(self, file_name):

        # Read from data set to a numpy array
        self.data_set_reg = np.genfromtxt(file_name, delimiter=',')
        self.entries_data = np.shape(self.data_set_reg)[0]
        self.len_data = np.shape(self.data_set_reg[0])[0]
        self.testing_data_set = None
        self.len_test_data = None

    # DISTANCE FORMULA FUNCTIONS
    # Calculates the euclidean distance between 2 vectors
    def distance_euclidean(self, c: "Int", target) -> "Double":
        return np.sqrt(np.sum(np.square(np.subtract(target[:self.len_data - 1], self.data_set_reg[c, :self.len_data - 1]))))

    # Calculates the manhattan distance between 2 vectors
    def distance_manhattan(self, c: "Int", target) -> "Double":
        return np.sum(np.abs(np.subtract(target[:self.len_data - 1], self.data_set_reg[c, :self.len_data - 1])))

    # FUNCTION SETTERS
    def _set_k(self, k_measure: "string") -> "None":
        switcher = {
            'static-3': 3,
            'static-10': 10,
            'static-15': 15,
            'squared': np.round(np.sqrt(self.entries_data)),
            'n-fold': np.round(self.entries_data / self.len_test_data) + 1
        }

        self.k = switcher.get(k_measure)

    # This method switches the distance calculating parameter depending on passing parameter.
    def _set_distance(self, distance: "string") -> "None":
        switcher = {
            'euclidean': self.distance_euclidean,
            'manhattan': self.distance_manhattan
        }
        self.applier_distance = switcher.get(distance)

    def value_regression(self, query_target):

        # Create a Series with indexes of data set to be calculated
        predicted_values = pd.Series(range(self.entries_data))
        # Calculate the distance between each entry in the data set and the query target
        distances_computed = predicted_values.apply(lambda f: self.applier_distance(f, query_target))

        # Obtain the indexes corresponding with the closest distances
        sorted_distances = np.argsort(distances_computed)

        # Calculate the formula for regression
        counted_distances = 0
        computed_distances = 0
        for i in sorted_distances[:self.k]:

            distance = 1/np.square(distances_computed[i])
            counted_distances += distance
            computed_distances += distance * self.data_set_reg[i, self.len_data - 1]

        return computed_distances / counted_distances

    def test_accuracy(self, file, k_set, distance):

        # Get the values of the testing data set
        self.testing_data_set = np.genfromtxt(file, delimiter=',')
        self.len_test_data = np.shape(self.testing_data_set)[0]

        # Set the K and distance to calculate
        self._set_k(k_set)
        self._set_distance(distance)

        # Get the average of the true values
        average_total = np.mean(self.testing_data_set[:, self.len_data - 1], dtype=np.float64)

        # Calculate the squared residuals and the sum of squares
        squared_residuals = 0
        sum_of_squares = 0

        for x in range(self.len_test_data):
            predicted_value = self.value_regression(self.testing_data_set[x])
            squared_residuals += np.square(predicted_value - self.testing_data_set[x, self.len_data - 1])
            sum_of_squares += np.square(average_total - self.testing_data_set[x, self.len_data - 1])

        # Final R squared
        r_squared = 1 - (squared_residuals / sum_of_squares)

        # Print results obtained
        print('The R squared is {:f}.'.format(r_squared))
        print('The parameters used are: K {0} and distance {1}.\n\n'.format(k_set, distance))

dert = KNN_regresion_Jose('regressionData/trainingData.csv')

for k in ['static-3', 'static-10', 'static-15', 'squared', 'n-fold']:
    for tech in ['euclidean', 'manhattan']:
        dert.test_accuracy('regressionData/testData.csv', k, tech)