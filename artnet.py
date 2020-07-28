"""
    Code adapted from https://github.com/natmourajr/LPSTutorials
    Contributors: @jonjoncardoso, @hellenlima

"""

import numpy as np
import pandas as pd

from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances

def calculate_distances(new_data_features, neurons_weights, distance_metric='euclidean'):
    """
    new_data_features: np.array containing features representing a new data point
    distance_metric: distance method used (default: euclidean)
    neurons: np.array containing all neurons weights of an ARTNet

    This method will return a Pandas Series with same size as `neurons`
        containing all distances between the neurons and the new datapoint
    """
    if distance_metric == 'euclidean':
        distances = euclidean_distances(neurons_weights,
                                        new_data_features.reshape(1,-1))
        return distances
    else:
        raise NotImplementedError('Distance metric "%s" not implemented' % distance_metric)


def get_min_distance(distances):
    id_winner_neuron = distances.argmin()
    min_dist = distances.min()
    return id_winner_neuron, min_dist


def calculate_winner_neuron_id(new_data, neurons, distance_metric, similarity_radius):
    """
    Parameters
    ----------
        new_data: array representing a new data point
        distance_metric: distance method used (default: euclidean)
        neurons (np.array): np.array containing all neurons of an ARTNet
    Return
    ----------
        id_winner_neuron (int): the id corresponding to the winner neuron for `new_data`
    """
    distances = calculate_distances(new_data_features=new_data,
                                    neurons_weights=neurons,
                                    distance_metric=distance_metric)
    id_winner_neuron, min_dist = get_min_distance(distances)
    if min_dist <= similarity_radius:
        return id_winner_neuron
    else:
        return -1


class ARTNet:
    """
    ARTNet class
    This class implements the Adaptive Resonance Theory
    """

    def __init__(self, similarity_radius=0.93,
                 distance_metric="euclidean", learning_rate=0.05,
                 neurons=[],
                 verbose=0,
                 prediction_col='prediction'):
        """
        ARTNet constructor
            similarity_radius: Similarity Radius (default: 0.1)
            distance_metric: distance method used (defaults: euclidean)
            learning_rate: rate representing how quickly a neuron abandons
                its old beliefs for new ones
            neurons: if it is not None, should be a pandas DataFrame containing
                pre-trained neurons
            calculate_metrics: choose wether we are going to calculate
                intra and inter cluster distances
        """

        super().__init__()
        self.similarity_radius = similarity_radius
        self.distance_metric = distance_metric
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.prediction_col = prediction_col

        self.neurons = neurons


    def create_neuron(self, new_data):
        """Add a new neuron to the network

        Parameters
        ----------
            new_data (np.array): an array of shape [1, n_features] representing new neuron weight to be created
        """
        if self.neurons == []:
            # No defined neurons yet
            self.neurons = new_data[np.newaxis, :]
        else:
            # Append new neuron to neurons array
            self.neurons = np.append(self.neurons, new_data[np.newaxis, :], axis=0)


    def _update_extra_cluster_dist(self, cl_id, cl_distances, cl_count=None):
        """ Not implemented in this version
        """
        raise NotImplementedError("Extra cluster distances calculation is not implemented. "
                                  "For now, choose calculate_metrics==False")


    def _calculate_updated_weights(self, winner_neuron, new_data):
        """
        The equation used to update the weights is the following:
          W(t+1) = a (W(t) - N) + N
        where :
            - a is the learning factor
            - W(t) holds the current weights
            - W(t+1) represents the new weights
            - N represents the new input data
        """
        # Update neuron weights
        learning_factor = (1 - self.learning_rate) ** 1

        distance = winner_neuron - new_data
        addend1 = distance * learning_factor 

        updated_weights = addend1 + new_data
        return updated_weights


    def update_winner_neuron(self, id_winner_neuron, new_data, new_data_dist, cl_distances):
        """Update winner neuron weights

        Parameters
        ----------
            id_winner_neuron (int)  : id of the winner neuron
            new_data (np.array)     : np.array containing the new data points
            new_data_dist (float)   : Euclidean distance of new_data to the winner neuron
        """
        # Get winner neuron weights
        winner_neuron = self.neurons[id_winner_neuron]

        updated_weights = self._calculate_updated_weights(winner_neuron, new_data)

        self.neurons[id_winner_neuron] = updated_weights


    def fit(self, X):
        """
        Implements training process

        Parameters
        ----------
            X: input training data of shape [n_samples, n_features]

        """
        for idx in range(X.shape[0]):
            if self.neurons == []:
                self.create_neuron(new_data=X[idx])
            else:
                distances = calculate_distances(new_data_features=X[idx],
                                        neurons_weights=self.neurons,
                                        distance_metric=self.distance_metric)
                id_winner_neuron, min_dist = get_min_distance(distances)
                if min_dist > self.similarity_radius:
                    self.create_neuron(new_data=X[idx])
                else:
                    self.update_winner_neuron(id_winner_neuron, X[idx], min_dist, distances)
        return self


    def partial_fit(self, data):
        return self.fit(data)


    def predict(self, data):
        """
        Parameters
        ----------
        data (np.array): data of shape [n_samples, n_features] containing new samples to be associated to neurons

        Returns
        -------
        list containing the id of the most similar neuron associated to each sample
        """

        distance_metric = self.distance_metric

        output = [calculate_winner_neuron_id(data[new_data],
                                             self.neurons,
                                             self.distance_metric,
                                             self.similarity_radius) for new_data in range(data.shape[0])]
        return output
