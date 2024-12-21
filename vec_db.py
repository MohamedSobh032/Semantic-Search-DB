from typing import Dict, List, Annotated
from ivf import ivf
import numpy as np
import os

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70
INDEX_PARAMS = {             # Number of centroids for each database size, could be changed. Could've made it dynamic but these numbers work well.
    "no_of_centroids":{
        1: 17,
        10: 10,
        15: 9,
        20: 3
    },
}

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", new_db = False, db_size = None, index_file_path: str = '.') -> None:
        '''
        Constructor for the VecDB class.

        Parameters:
        ------------
        database_file_path: str
            The path to the database file. Default is "saved_db.dat".
        new_db: bool
            If True, a new database is generated. Default is False.
        db_size: int
            The size of the database to be generated. Default is None.
        index_file_path: str
            The path to the index file. Default is '.'.
        '''
        # INITIALIZE VARIABLES
        self.db_path = database_file_path
        # Getting which database size is being used.
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

        # Initialize IVF object.
        self.ivf = ivf(database_file_path)

        # Set the index file path.
        self.index_file_path = index_file_path

        # If new_db is set to false, then the database is loaded from the file. Otherwise, a new database is generated.
        if new_db:
            # No answer provided.
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            # Make the directory if it doesn't exist.
            os.makedirs(self.index_file_path, exist_ok=True)
            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        '''
        Generates a new database.

        Parameters:
        ------------
        size: int
            The size of the database to be generated.
        '''
         # Initialize a random number generator with the seed number.
        rng = np.random.default_rng(DB_SEED_NUMBER)
        # Generate a matrix of random vectors.
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        # Write the vectors to the database file.
        self._write_vectors_to_file(vectors)
        print("Database generated.")
        # Build the index.
        self._build_index(self.index_file_path, vectors)

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        """
        Writes a numpy array of vectors to a file using memory mapping for efficient storage.

        Parameters:
        ------------
        vectors: np.ndarray
            The array of vectors to be written to the file.
        """
        # Create a memory-mapped file for the vectors.
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        # Copy the vectors to the memory-mapped file.
        mmap_vectors[:] = vectors[:]
        # Flush the memory-mapped file to ensure that the data is written to disk.
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        '''
        Get the number of records in the database.

        Returns:
        ------------
        int
            The number of records in the database.
        '''
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        '''
        Insert records into the database.

        Parameters:
        ------------
        rows: np.ndarray
            The array of vectors to be inserted into the database.
        '''
        # Get the number of records in the database.
        num_old_records = self._get_num_records()
        # Calculate the new shape of the memory-mapped file.
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        # Open the memory-mapped file in read-write mode.
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        # Flush the memory-mapped file to ensure that the data is written to disk.
        mmap_vectors.flush()
        #TODO: might change to call insert in the index, if you need
        self._build_index(self.index_file_path, rows)

    def get_one_row(self, row_num: int) -> np.ndarray:
        '''
        Get one row from the database.

        Parameters:
        ------------
        row_num: int
            The row number to be retrieved.
        
        Returns:
        ------------
        np.ndarray
            The vector at the specified row number.
        '''
        try:
            # Calculate the offset of the row in the memory-mapped file.
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        # Handle any exceptions that occur during the process.
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        '''
        Get all rows from the database.

        Returns:
        ------------
        np.ndarray
            The array of vectors in the database.
        '''
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        # Use a memory-mapped file to access the database file.
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        """
        Retrieves the top-k most similar vectors to the query from the index.

        Parameters:
        ------------
        query : Annotated[np.ndarray, (1, DIMENSION)]
            The query vector of shape (1, DIMENSION) used for similarity search.
        top_k : int, optional
            The number of closest vectors to retrieve. Defaults to 5.

        Returns:
        ------------
        list
            A list of the top-k most similar vectors to the query, ranked by similarity.
        """
        # Get the number of centroids to use for the search.
        no_of_centroids = INDEX_PARAMS["no_of_centroids"][self.size]
        # Perform the similarity search using the IVF index.
        results = self.ivf.find_nearest(self.index_file_path, query, top_k, no_of_centroids)
        return results
    
    def _cal_score(self, vec1, vec2):
        """
        Calculate the cosine similarity between two vectors.

        Parameters:
        ------------
        vec1 : np.ndarray
            The first vector.
        vec2 : np.ndarray
            The second vector.
        
        Returns:
        ------------
        float
            The cosine similarity between the two vectors.
        """
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self, path, data = None):
        """
        Build the index for the database.

        Parameters:
        ------------
        path : str
            The path to the index file.
        data : np.ndarray, optional
            The data to be indexed. Defaults to None.
        """
        self.ivf.build_index(path, data)
