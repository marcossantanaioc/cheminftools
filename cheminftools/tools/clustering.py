from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from kneed import KneeLocator
from sklearn.cluster import KMeans

from cheminftools.tools.featurizer import MolFeaturizer


class BaseClustering:
    def __init__(self, dataset: List[str], descriptor_name: str = 'morgan', descriptor_params={}):
        self.dataset = dataset
        self.descriptor_name = descriptor_name
        self.descriptor_params = descriptor_params

    @property
    def featurizer(self):
        return MolFeaturizer(descriptor_type=self.descriptor_name, params=self.descriptor_params)

    def cluster(self):
        pass


class KMeansClustering(BaseClustering):
    """
    Performs k-means clustering on a dataset of molecules.

    Attributes
    ----------
    dataset
        An array of features with shape (n,p), where n is the number of molecules and p is the number of descriptors.
    """

    def __init__(self, dataset: List[str]):
        super().__init__(dataset=dataset)
        self.X = self.featurizer.transform(dataset)

    def cluster(self,
                n_clusters: int = 10,
                max_iter: int = 5,
                n_init: int = 5,
                init: str = 'k-means++',
                random_state=None):

        """
        Run k-means on the dataset

        Parameters
        ----------
        n_clusters
            Number of clusters

        max_iter
            Maximum number of iterations of the k-means algorithm for a single run.
        n_init
            Number of times the k-means algorithm is run with different centroid seeds.
        init
            ‘k-means++’ : selects initial cluster centroids using sampling based on an empirical
            probability distribution of the points’ contribution to the overall inertia.
            This technique speeds up convergence. The algorithm implemented is “greedy k-means++”.
            It differs from the vanilla k-means++ by making several trials at each sampling step and
            choosing the best centroid among them.

            ‘random’: choose n_clusters observations (rows) at random from data for the initial centroids.

            If an array is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.

            If a callable is passed, it should take arguments X, n_clusters
            and a random state and return an initialization.
        random_state
            Determines random number generation for centroid initialization.
            Use an int to make the randomness deterministic.

        Returns
        -------
        labels
            Clustering labels
        """

        cls = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, random_state=random_state)
        cls.fit(self.X)

        self.clusterer = cls
        self.labels = cls.labels_
        return self.labels

    def elbow_method(self, n_clusters: List, figsize: Tuple = (12, 9), **kwargs):

        self.inertias = []
        for n in n_clusters:
            self.cluster(n, **kwargs)
            inertia = self.clusterer.inertia_
            self.inertias.append(inertia)

        # Find elbow
        params = {"curve": "convex",
                  "direction": "decreasing"}

        knee_finder = KneeLocator(n_clusters, self.inertias, **params)
        self.elbow_value = knee_finder.elbow

        # Plot Elbow
        sns.set_context('paper', font_scale=2.0)
        sns.set_style('whitegrid')

        fig = plt.figure(figsize=figsize)
        ax = sns.lineplot(x=n_clusters, y=np.array(self.inertias), linewidth=2.5, marker='o', color='blue',
                          markersize=7)

        ax.set_xlabel('Number of clusters (K)')
        ax.set_ylabel('Distortion')
        sns.despine(right=True, top=True)
        plt.title('K-means Elbow method', fontweight='bold', fontsize=22)

        if self.elbow_value is not None:
            elbow_label = "Elbow at $K={}$".format(self.elbow_value)
            ax.axvline(self.elbow_value, c='k', linestyle="--", label=elbow_label)
            ax.legend(loc="best", fontsize=18, frameon=True)
        for i in ax.spines.items():
            i[1].set_linewidth(1.5)
            i[1].set_color('k')

        plt.tight_layout()
        plt.show()
