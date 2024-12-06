import numpy as np

class hnsw:
    
    def __init__(self):
        '''
        Initalization of the hnsw structure
        '''
        pass

    # cosine similarity
    def get_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        '''
        Gets the angle between two vector
        param v1: vector 1
        param v2: vector 2
        '''
        return 1.0 - (np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    # insert

    # select

    # find nearest
    def find_nearest_neighbours_in_layer(self, vector):
        pass