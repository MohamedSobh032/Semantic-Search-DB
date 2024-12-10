# Modified ivf.py
import numpy as np
import pickle
import os
from sklearn import preprocessing
import nanopq
import math

from sklearn.cluster import MiniBatchKMeans

class ivf:
    def __init__(self, clusters: int = 10000, batches: int = 100000, n_subvectors: int = 8):
        self.batch_size = batches
        self.cluster_num = clusters
        # Calculate optimal M (number of subvectors) to handle any dimension
        self.M = n_subvectors
        self.index = None
        self.centroids = None
        
    def _pad_vectors(self, vectors: np.ndarray) -> tuple[np.ndarray, int]:
        """Pad vectors to make dimension divisible by M"""
        d = vectors.shape[1]
        if d % self.M != 0:
            pad_size = self.M - (d % self.M)
            padded = np.pad(vectors, ((0, 0), (0, pad_size)), 'constant')
            return padded, pad_size
        return vectors, 0

    def build_index(self, path: str, data = None) -> None:
        if data is None:
            vectors = []
            ids = []
            idx = 0
            while batch := self.load_batch(index=idx):
                vectors.extend(list(batch.values()))
                ids.extend(list(batch.keys()))
                idx += 1
            vectors = np.array(vectors, dtype=np.float32)
        else:
            vectors = np.array(data, dtype=np.float32)
            ids = list(range(len(vectors)))

        # Pad vectors if needed
        padded_vectors, pad_size = self._pad_vectors(vectors)
        
        # Initialize and train OPQ
        self.opq = nanopq.OPQ(M=self.M, Ks=256)
        self.opq.fit(vecs=padded_vectors, pq_iter=20, rotation_iter=10)

        # Coarse clustering
        normalized = preprocessing.normalize(vectors)
        kmeans = MiniBatchKMeans(self.cluster_num, batch_size=self.batch_size)
        cluster_labels = kmeans.fit_predict(normalized)
        self.centroids = kmeans.cluster_centers_

        # Build single index file
        self.index = {
        'metadata': {
            'n_clusters': self.cluster_num,
            'n_subvectors': self.M,
            'dimension': vectors.shape[1],
            'pad_size': pad_size
        },
        'centroids': self.centroids,
        'opq_params': {
            'codewords': self.opq.codewords,
            'rotation_matrix': self.opq.R  # Changed to use R
        },
        'clusters': {}
        }

        # Encode vectors per cluster
        for cluster_id in range(self.cluster_num):
            mask = cluster_labels == cluster_id
            if np.any(mask):
                cluster_vectors = padded_vectors[mask]
                cluster_ids = np.array(ids)[mask]
                codes = self.opq.encode(cluster_vectors)
                self.index['clusters'][cluster_id] = {
                    'codes': codes,
                    'ids': cluster_ids
                }

        # Save single index file
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'ivf_index.pkl'), 'wb') as f:
            pickle.dump(self.index, f)
    # Add to ivf class in ivf.py
    def get_similarity(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between query and vectors
        Args:
            query: normalized query vector (1, d)
            vectors: normalized vectors to compare against (n, d)
        Returns:
            similarities: cosine similarities (n,)
        """
        return np.dot(query, vectors.T)

    def find_nearest(self, path: str, query: np.ndarray, no_of_matches: int, no_of_centroids: int) -> list[int]:
        # Load index if needed
        if self.index is None:
            with open(os.path.join(path, 'ivf_index.pkl'), 'rb') as f:
                self.index = pickle.load(f)
                self.centroids = self.index['centroids']
                # Reconstruct OPQ
                self.opq = nanopq.OPQ(M=self.index['metadata']['n_subvectors'])
                self.opq.codewords = self.index['opq_params']['codewords']
                self.opq.R = self.index['opq_params']['rotation_matrix']
        
        # First normalize original query (keep as 2D for similarity computation)
        query_2d = query.reshape(1, -1).astype(np.float32)
        query_normalized = preprocessing.normalize(query_2d)
        
        # Find nearest centroids using original dimensions
        sims = self.get_similarity(query_normalized, self.centroids)
        nearest_clusters = np.argsort(sims.flatten())[-no_of_centroids:][::-1]

        # Convert to 1D and pad query for PQ distance computation
        query_1d = query_2d.flatten()  # Convert to 1D for nanopq
        if self.index['metadata']['pad_size'] > 0:
            query_1d = np.pad(query_1d, (0, self.index['metadata']['pad_size']), 'constant')

        # Search in selected clusters using 1D query
        candidates = []
        dt = self.opq.dtable(query=query_1d)  # Pass 1D vector
        
        for cluster_id in nearest_clusters:
            if cluster_id in self.index['clusters']:
                cluster_data = self.index['clusters'][cluster_id]
                codes = cluster_data['codes']
                ids = cluster_data['ids']
                dists = dt.adist(codes=codes)
                candidates.extend(zip(ids, -dists))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in candidates[:no_of_matches]]