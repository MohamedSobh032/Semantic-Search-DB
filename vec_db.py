from typing import Dict, List, Annotated
from ivf import ivf
import numpy as np
import os

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70
INDEX_PARAMS = {
    "no_of_centroids":{
        1: 17,
        10: 11,
        15: 10,
        20: 4
    },
}

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", new_db = False, db_size = None, index_file_path: str = '.') -> None:
        
        # INITIALIZE VARIABLES
        self.db_path = database_file_path
        if index_file_path == "1M_Index":
            self.size = 1
        elif index_file_path == "10M_Index":
            self.size = 10
        elif index_file_path == "15M_Index":
            self.size = 15
        elif index_file_path == "20M_Index":
            self.size = 20
        else:
            self.size = 1
        
        print(f"size : {self.size}")

        self.ivf = ivf(database_file_path)
        self.index_file_path = index_file_path

        # IF NEW DATABASE NEEDED
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            os.makedirs(self.index_file_path, exist_ok=True)
            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        print("Database generated")
        self._build_index(self.index_file_path, vectors)

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        #TODO: might change to call insert in the index, if you need
        self._build_index(self.index_file_path, rows)

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        no_of_centroids = INDEX_PARAMS["no_of_centroids"][self.size]
        results = self.ivf.find_nearest(self.index_file_path, query, top_k, no_of_centroids)
        return results
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self, path, data = None):
        self.ivf.build_index(path, data)
