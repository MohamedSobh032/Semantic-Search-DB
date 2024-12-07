import numpy as np      # Maths operations
import pickle           # Serializations of data
from sklearn import preprocessing
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.cluster.vq import whiten, kmeans, vq, kmeans2
import os
import gc
import math

class ivf:
    def __init__(self, clusters: int = 1000, batches: int = 100000) -> None:
        '''
        Constructor of IVF
        '''
        self.batch_size = batches
        self.cluster_num = clusters
        self.kmeans = KMeans(n_clusters=clusters)
        self.inverted_index = {}

    # READ FILE WITH INDEX
    def load_batch(self, index: int):
        '''
        reads data file with index as binary
        param index: which data file to read
        '''
        file_name = f'data/data_{index}.pkl'
        with open(file_name, 'rb') as file:
            return pickle.load(file)
        return None
    
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
        '''
        Builds the index of the data
        param data: data to build the index
        '''
        index = [{} for i in range(self.cluster_num)]
        if data is None:
            # cluster the data into batches
            kmeans = MiniBatchKMeans(self.cluster_num, random_state=0, batch_size=self.batch_size, max_iter=10, init_size='auto')
            idx = 0
            while batch := self.load_batch(index=idx):
                idx += 1
                np_batch = np.array(list(batch.values()))
                for i in range(0, len(np_batch), self.batch_size):
                    kmeans.partial_fit(preprocessing.normalize(np_batch[i : i + self.batch_size]))
                    for j, k in enumerate(kmeans.labels_):
                        index[k][idx * len(np_batch) + j + i] = batch[idx * len(np_batch) + j + i]
                del batch
                del np_batch
                gc.collect()
            centroids = kmeans.cluster_centers_
        else:
            # turn data to array as i have no need for keys
            new_data = np.array(list(data))
            if len(new_data) > 1000000:
                kmeans = MiniBatchKMeans(self.cluster_num, random_state=0, batch_size=self.batch_size,max_iter=10,n_init="auto")
                for i in range(0,len(new_data),self.batch_size):
                    kmeans.partial_fit(preprocessing.normalize(new_data[i:i+self.batch_size]))
                    indices = kmeans.labels_
                    for j,k in enumerate(indices):
                        index[k][j+i] = new_data[j+i]
                centroids = kmeans.cluster_centers_
            else:
                centroids,cluster_indices = kmeans2(preprocessing.normalize(new_data), self.cluster_num)
                for i,key in enumerate(cluster_indices):
                    index[key][i] = data[i]
            del new_data
            del data
        for i in range(self.cluster_num): #make files for each cluster
            file_name = path + f'/data_{i}.pkl'
            with open(file_name, 'wb') as file:
                pickle.dump(index[i], file)
        centroids_file = path + '/centroids.pkl'
        with open(centroids_file, 'wb') as file:
            pickle.dump(centroids, file)
        del index
        del centroids
        gc.collect()
        print('Finished Indexing')
        
    # search to find nearest index
    def find_nearest(self, path: str, query: np.ndarray, centroids: np.ndarray, no_of_matches: int, no_of_centroids: int) -> list[int]:
        '''
        Finds the nearest neighbors for a query vector
        '''
        # Convert query to numpy and normalize 
        query = np.array(query).reshape(1, -1)
        query = preprocessing.normalize(query)
        
        # Get similarities to centroids
        sims = self.get_similarity(query, centroids)
        sims = sims.flatten() # Ensure 1D array
        
        # Get indices of nearest centroids (highest similarity)
        nearest_index = np.argsort(sims)[-no_of_centroids:][::-1]
        
        # Store candidates and their similarities
        candidates = []
        
        # Search in nearest clusters
        for cluster_id in nearest_index:
            # Convert cluster_id to integer scalar
            cluster_id_int = int(cluster_id)
            points = self.load_file(f'{path}/data_{cluster_id_int}.pkl')
            
            if points:
                for point_idx, point_vec in points.items():
                    sim = self.get_similarity(point_vec, query)
                    candidates.append((point_idx, point_vec, float(sim)))
                
                # Clear memory
                del points
                gc.collect()
        
        # Sort by similarity (descending)
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Return top matches
        return [x[0] for x in candidates[:no_of_matches]]
    

#TO DO: 
# 1. Add Product Quantization
# 2. Make Index one file
# 3. Find clusters sweet spot
# 4. Run on Kaggle