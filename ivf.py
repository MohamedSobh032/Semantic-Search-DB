import numpy as np      # Maths operations
import pickle           # Serializations of data
from productquantization import ProductQuantizer
from sklearn import preprocessing
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.cluster.vq import whiten, kmeans, vq, kmeans2
import os
import gc
import math

class ivf:
    def __init__(self, clusters: int = 8192, batches: int = 500000, n_subvectors: int = 16, n_clusters_pq: int = 256) -> None:
        self.batch_size = batches
        self.cluster_num = clusters
        self.pq = ProductQuantizer(n_subvectors, n_clusters_pq)
        # Single index structure:
        # {cluster_id: {"codes": encoded_vectors, "ids": original_ids}}
        self.index = {}
        self.centroids = None
    
    # READ FILE WITH FILENAME
    def load_file(self, file_name: str):
        '''
        reads data file as binary
        param file_name: which data file to read
        '''
        with open(file_name, 'rb') as file:
            return pickle.load(file)
        return None

    # INSERT
    def append_to_file(self, filename: str, idx: int, data: np.ndarray) -> None:
        '''
        inserts new data to a file
        param filename: filename
        param data: vector to be inserted
        '''
        with open(filename, 'a', newline = '') as file:
            arr = ','.join(map(str, data))
            file.write(f'{idx},{arr}\n')
    
    # DELETE ALL DATA INSIDE A DIRECTORY
    def delete_file_data(self, dir_path: str):
        '''
        Deletes all data files inside a directory
        param dir_path: path to the directory to delete
        '''
        for file in os.listdir(dir_path):
            if file.endswith('.pkl'):
                os.remove(os.path.join(dir_path, file))
    
    # Cosine Similarity
    def get_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        '''
        Gets the angle between two vector
        param v1: vector 1
        param v2: vector 2
        '''
        return np.dot(v1, v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    # build index
    def build_index(self, path: str, data = None) -> None:
    # First pass: Cluster assignment
        vectors = data
        ids = list(range(len(vectors)))

        # Normalize and cluster
        normalized = preprocessing.normalize(vectors)
        kmeans = MiniBatchKMeans(self.cluster_num, batch_size=self.batch_size)
        cluster_labels = kmeans.fit_predict(normalized)
        self.centroids = kmeans.cluster_centers_

        # Initialize clusters
        for i in range(self.cluster_num):
            self.index[i] = {"codes": [], "ids": []}

        # Second pass: PQ encoding per cluster
        for cluster_id in range(self.cluster_num):
            cluster_mask = cluster_labels == cluster_id
            cluster_vectors = vectors[cluster_mask]
            cluster_ids = np.array(ids)[cluster_mask]
            
            if len(cluster_vectors) > 0:
                self.pq.fit(cluster_vectors)
                codes = self.pq.encode(cluster_vectors)
                self.index[cluster_id]["codes"] = codes
                self.index[cluster_id]["ids"] = cluster_ids

        # Save single index file
        with open(os.path.join(path, 'ivf_index.pkl'), 'wb') as f:
            pickle.dump({
                'index': self.index,
                'centroids': self.centroids,
                'pq_params': {
                    'M': self.pq.M,
                    'K': self.pq.K,
                    'original_d': self.pq.original_d
                }
            }, f)
        print("Finished Indexing.")
        
    # search to find nearest index
    def find_nearest(self, path: str, query: np.ndarray, no_of_matches: int, no_of_centroids: int) -> list[int]:
        # Load index if not in memory
        if not hasattr(self, 'index') or self.index is None:
            with open(os.path.join(path, 'ivf_index.pkl'), 'rb') as f:
                data = pickle.load(f)
                self.index = data['index']
                self.centroids = data['centroids']

        query = preprocessing.normalize(query.reshape(1, -1))
        sims = self.get_similarity(query, self.centroids)
        nearest_clusters = np.argsort(sims.flatten())[-no_of_centroids:][::-1]
        
        candidates = []
        for cluster_id in nearest_clusters:
            codes = self.index[cluster_id]["codes"]
            ids = self.index[cluster_id]["ids"]
            
            if len(codes) > 0:
                vectors = self.pq.decode(codes)
                sims = self.get_similarity(query, vectors)
                for idx, sim in zip(ids, sims.flatten()):
                    candidates.append((idx, float(sim)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in candidates[:no_of_matches]]
    
#TO DO: 
# 1. Add Product Quantization
# 2. Make Index one file
# 3. Find clusters sweet spot
# 4. Run on Kaggle