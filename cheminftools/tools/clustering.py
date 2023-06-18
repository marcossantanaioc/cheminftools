from typing import List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from sklearn.cluster import KMeans


class KMeansClustering(BaseClustering):
    """
    Performs k-means clustering on a dataset of molecules.

    Attributes
    ----------

    dataset
        An array of features with shape (n,p), where n is the number of molecules and p is the number of descriptors.

    """

    def __init__(self, dataset: ArrayLike):

        self.dataset = dataset

    def cluster(self, n_clusters: int = 10, **kwargs):

        """
        Run k-means on the dataset

        Parameters
        ----------
        n_clusters
            Number of clusters

        Other Parameters
        -----------------
        max_iter : int (default=5)
        n_init : int (default=5)
        init : str (default='k-means++')
        random_state : int (default=None)

        Returns
        -------
        labels
            Clustering labels
        """

        max_iter = kwargs.get('max_iter', 500)
        n_init = kwargs.get('n_init', 10)
        init = kwargs.get('init', 'k-means++')
        random_state = kwargs.get('random_state', None)

        cls = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, random_state=random_state)
        cls.fit(self.dataset)

        self._clusterer = cls
        self._labels = cls.labels_
        return self._labels

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
