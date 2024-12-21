import numpy as np
import pickle           # Serializations of data
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.vq import kmeans2
import os
import gc
DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class ivf:
    def __init__(self, database_file_path="saved_db.dat", clusters: int = 500, sub_cluster: int = 50, batches: int = 100000) -> None:
        """
        Initializes an IVF (Inverted File) class instance for clustering and efficient data retrieval.

        Parameters:
        ----------
        database_file_path : str, optional
            The path to the file where the database is stored or will be saved. Defaults to "saved_db.dat".
        clusters : int, optional
            The number of clusters for the first level of clustering.
            This determines the initial partitioningof the data space. Defaults to 500.
        sub_cluster : int, optional
            The number of clusters for the second level of clustering.
            This allows more precise data retrieval. Defaults to 50.
        batches : int, optional
            The batch size for MiniBatchKMeans, which controls the number of samples processed at a time 
            during clustering. Defaults to 100000.
        """
        self.batch_size = batches
        self.cluster_num = clusters
        self.database_file_path = database_file_path
        self.sub_cluster_num = sub_cluster


    # Cosine Similarity
    def get_similarity(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        '''
        Computes the cosine similarity between two vectors or between a vector and a matrix of vectors

        Parameters:
        ----------
        v1 : np.ndarray
            The first vector or matrix of vectors
        v2 : np.ndarray
            The second vector or matrix of vectors
        '''
        v1_norm = np.linalg.norm(v1, axis=1, keepdims=True)
        v2_norm = np.linalg.norm(v2, axis=1, keepdims=True)
        return np.dot(v1, v2.T) / (v1_norm * v2_norm.T)
    
    # build index
    def build_index(self, path: str, data=None) -> None:
        '''
        Builds a hierarchical index of the data using multiple levels of clustering
        param data: data to build the index

        Parameters:
        ----------
        path : str
            The path to the directory where the index files will be stored.
        data : np.ndarray, optional
            The data to be indexed. If not provided, the data will be loaded from the database file.
        '''
        new_data = np.array(list(data))
        
        # Remove rows with NaN or inf values
        new_data = new_data[~np.isnan(new_data).any(axis=1)]
        new_data = new_data[~np.isinf(new_data).any(axis=1)]
        
        normalized_data = preprocessing.normalize(new_data)
        
        # Initialize structures
        hierarchy = {}
        level_centroids = {}
        
        # First level clustering
        if len(new_data) >= 1000000:
            print('Using MiniBatchKMeans for first level clustering')
            # Use MiniBatchKMeans for large datasets by choosing a random subset of the data at each iteration.
            kmeans_l1 = MiniBatchKMeans(self.cluster_num, random_state=0, 
                                        batch_size=self.batch_size, max_iter=10, n_init="auto")
            # 
            kmeans_l1.fit(normalized_data)
            l1_centroids = kmeans_l1.cluster_centers_
            l1_labels = kmeans_l1.labels_
            print('First level clustering complete')
        else:
            # Use kmeans2 for small datasets which takes the entire dataset at once.
            l1_centroids, l1_labels = kmeans2(normalized_data, self.cluster_num)
        
        # Store first level centroids
        level_centroids[0] = l1_centroids
        level_centroids[1] = {}
        # Second level clustering
        for i in range(self.cluster_num):
            print(f'Clustering for first level cluster {i}')
            cluster_data = normalized_data[l1_labels == i]
            
            # Remove rows with NaN or inf values in cluster_data.
            cluster_data = cluster_data[~np.isnan(cluster_data).any(axis=1)]
            cluster_data = cluster_data[~np.isinf(cluster_data).any(axis=1)]
            
            # Printing the shape of the cluster data for debugging purposes.
            print(f'Cluster {i} data shape: {cluster_data.shape}')  
            if len(cluster_data) == 1:
                # If the cluster contains only one vector, use it as the centroid
                l2_centroids = cluster_data
                l2_labels = np.array([0])
            # If the cluster contains more than the batch size, use MiniBatchKMeans.
            elif len(cluster_data) > self.batch_size:
                kmeans_l2 = MiniBatchKMeans(self.sub_cluster_num, random_state=0, 
                                            batch_size=self.batch_size, max_iter=10, n_init="auto")
                kmeans_l2.fit(cluster_data)
                l2_centroids = kmeans_l2.cluster_centers_
                l2_labels = kmeans_l2.labels_
            else:
                # Use kmeans2 for small datasets which takes the entire dataset at once.
                l2_centroids, l2_labels = kmeans2(cluster_data, max(2, self.sub_cluster_num))
            
            # Store second level centroids.
            level_centroids[1][i] = l2_centroids
            
            # Create leaf nodes with data indices.
            hierarchy[i] = {}
            cluster_indices = np.where(l1_labels == i)[0]
            for j in range(len(l2_centroids)):
                sub_cluster_indices = cluster_indices[l2_labels == j]
                hierarchy[i][j] = sub_cluster_indices.tolist()

                # Dump the leaf nodes to disk.
                with open(os.path.join(path, f'ivf_hierarchy_{i}_{j}.pkl'), 'wb') as f:
                    pickle.dump(hierarchy[i][j], f)
            # Print completion message for each first level cluster.
            print(f'Clustering for first level cluster {i} complete')
        
        # Dump the first level centroids to disk.
        with open(os.path.join(path, 'ivf_centroids.pkl'), 'wb') as f:
            pickle.dump(level_centroids, f)
        
        # Cleanups
        del new_data
        del normalized_data
        del hierarchy
        del level_centroids
        gc.collect()
        
        print('Finished Indexing')
        
    # search to find nearest index
    def find_nearest(self, path: str, query: np.ndarray, no_of_matches: int, no_of_centroids: int) -> list[int]:
        '''
        Finds the nearest neighbors for a query vector.

        Parameters:
        ----------
        path : str
            The path to the directory where the index files are stored.
        query : np.ndarray
            The query vector for which the nearest neighbors are to be found.
        no_of_matches : int
            The number of nearest neighbors to be found.
        no_of_centroids : int
            The number of centroids to be considered at the second level of the index.
        '''
        # Convert query to numpy and normalize 
        query = np.array(query).reshape(1, -1)
        query = preprocessing.normalize(query)
        
        # Load first level centroids
        with open(os.path.join(path, 'ivf_centroids.pkl'), 'rb') as f:
            index_centroids = pickle.load(f)

        # Get first level centroids.
        first_level_centroids = index_centroids[0]
        
        # Get similarities to first level centroids
        sims_l1 = self.get_similarity(query, first_level_centroids)
        sims_l1 = sims_l1.flatten() # Ensure 1D array
        
        # Get indices of nearest first level centroids (highest similarity)
        nearest_l1_index = np.argsort(sims_l1)[-no_of_centroids:][::-1]
        
        # Store candidates and their similarities
        candidates = []
        
        # Search in nearest first level clusters
        for l1_cluster_id in nearest_l1_index:
            l1_cluster_id_int = int(l1_cluster_id)
            
            # Load second level centroids for the current first level cluster
            second_level_centroids = index_centroids[1][l1_cluster_id_int]
            
            # Check if the second level centroids contain only one vector
            if len(second_level_centroids) == 1:
                with open(os.path.join(path, f'ivf_hierarchy_{l1_cluster_id_int}_0.pkl'), 'rb') as f:
                    data_indices = pickle.load(f)
                for data_index in data_indices:
                    data_vector = self.get_one_row(data_index)
                    sim = self.get_similarity(query, data_vector.reshape(1, -1))
                    candidates.append((data_index, data_vector, float(sim)))
            else:
                # Get similarities to second level centroids
                sims_l2 = self.get_similarity(query, second_level_centroids)
                sims_l2 = sims_l2.flatten() # Ensure 1D array
                
                # Get indices of nearest second level centroids (highest similarity)
                nearest_l2_index = np.argsort(sims_l2)[-no_of_centroids:][::-1]
                
                # Collect data indices from nearest second level clusters
                for l2_cluster_id in nearest_l2_index:
                    l2_cluster_id_int = int(l2_cluster_id)
                    with open(os.path.join(path, f'ivf_hierarchy_{l1_cluster_id_int}_{l2_cluster_id_int}.pkl'), 'rb') as f:
                        data_indices = pickle.load(f)
                    # Retrieve actual data vectors and compute similarities
                    for data_index in data_indices:
                        data_vector = self.get_one_row(data_index)
                        sim = self.get_similarity(query, data_vector.reshape(1, -1))
                        candidates.append((data_index, data_vector, float(sim)))
        # Sort by similarity (descending)
        candidates.sort(key=lambda x: x[2], reverse=True)
        # Return the top k nearest vectors
        return [candidate[0] for candidate in candidates[:no_of_matches]]
    
    def get_one_row(self, row_num: int) -> np.ndarray:
        '''
        Loads a single row from the database file.

        Parameters:
        ----------
        row_num : int
            The row number of the data to be loaded.
        '''
        try:
            # Calculate the offset of the row in the file.
            offset = row_num * DIMENSION * ELEMENT_SIZE
            # Use memory-mapped file to access the database.
            mmap_vector = np.memmap(self.database_file_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            # Return the row as a numpy array.
            return np.array(mmap_vector[0])
        # Handle non-existent row numbers
        except Exception as e:
            return f"An error occurred: {e}"
