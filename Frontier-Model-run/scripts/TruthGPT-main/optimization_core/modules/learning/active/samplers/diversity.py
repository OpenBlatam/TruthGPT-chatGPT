"""
Diversity Sampler
=================

Implements diversity-based sampling using clustering and nearest neighbors.
"""
import numpy as np
import logging
from typing import Optional
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from ..config import ActiveLearningConfig

logger = logging.getLogger(__name__)

class DiversitySampler:
    """Diversity-based sampling implementation."""
    
    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self.diversity_history = []
        logger.info("✅ Diversity Sampler initialized")
    
    def sample_diverse(self, unlabeled_data: np.ndarray, labeled_data: np.ndarray = None,
                      n_samples: int = None) -> np.ndarray:
        """Query diverse points from unlabeled pool."""
        logger.info(f"🎯 Sampling diverse points using method: {self.config.diversity_method}")
        
        if n_samples is None:
            n_samples = self.config.n_query_samples
        
        method = self.config.diversity_method
        if method == "kmeans":
            diverse_samples = self._kmeans_diversity_sampling(unlabeled_data, n_samples)
        elif method == "nearest_neighbors":
            diverse_samples = self._nearest_neighbors_diversity_sampling(unlabeled_data, labeled_data, n_samples)
        elif method == "clustering":
            diverse_samples = self._clustering_diversity_sampling(unlabeled_data, n_samples)
        else:
            diverse_samples = self._kmeans_diversity_sampling(unlabeled_data, n_samples)
        
        self.diversity_history.append({
            'method': method,
            'n_samples': n_samples,
            'selected_samples': diverse_samples
        })
        
        return diverse_samples
    
    def _kmeans_diversity_sampling(self, data: np.ndarray, n_samples: int) -> np.ndarray:
        """Sample points from different K-means clusters."""
        kmeans = KMeans(n_clusters=self.config.n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        
        diverse_samples = []
        samples_per_cluster = max(1, n_samples // self.config.n_clusters)
        
        for cluster_id in range(self.config.n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                count = min(len(cluster_indices), samples_per_cluster)
                selected_indices = np.random.choice(cluster_indices, count, replace=False)
                diverse_samples.extend(data[selected_indices])
        
        # Trim or pad to exact n_samples
        if len(diverse_samples) > n_samples:
            indices = np.random.choice(len(diverse_samples), n_samples, replace=False)
            diverse_samples = [diverse_samples[i] for i in indices]
        elif len(diverse_samples) < n_samples:
            remaining = n_samples - len(diverse_samples)
            # Add random ones from full data to fill
            random_indices = np.random.choice(len(data), remaining, replace=False)
            diverse_samples.extend(data[random_indices])
            
        return np.array(diverse_samples)
    
    def _nearest_neighbors_diversity_sampling(self, unlabeled_data: np.ndarray, 
                                            labeled_data: np.ndarray, n_samples: int) -> np.ndarray:
        """Sample points most distant from labeled set."""
        if labeled_data is None or len(labeled_data) == 0:
            indices = np.random.choice(len(unlabeled_data), n_samples, replace=False)
            return unlabeled_data[indices]
        
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(labeled_data)
        distances, _ = nn.kneighbors(unlabeled_data)
        farthest_indices = np.argsort(distances.flatten())[-n_samples:]
        return unlabeled_data[farthest_indices]
    
    def _clustering_diversity_sampling(self, data: np.ndarray, n_samples: int) -> np.ndarray:
        """Cluster into n_samples and pick centers."""
        kmeans = KMeans(n_clusters=n_samples, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        
        diverse_samples = []
        for cluster_id in range(n_samples):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                center = kmeans.cluster_centers_[cluster_id]
                # Pick point closest to centroid
                dists = np.linalg.norm(data[cluster_indices] - center, axis=1)
                diverse_samples.append(data[cluster_indices[np.argmin(dists)]])
        
        return np.array(diverse_samples)

