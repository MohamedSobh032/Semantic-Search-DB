import numpy as np
from sklearn.cluster import MiniBatchKMeans

class ProductQuantizer:
    def __init__(self, n_subvectors: int, n_clusters: int, batch_size: int = 10000):
        self.M = n_subvectors
        self.K = n_clusters
        self.batch_size = batch_size
        self.codebooks = []
        self.d = None

    def fit(self, X: np.ndarray):
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on empty array")
            
        self.original_d = X.shape[1]
        actual_k = min(self.K, X.shape[0])
        
        # Handle padding
        if self.original_d % self.M != 0:
            pad_size = self.M - (self.original_d % self.M)
            self.d = self.original_d + pad_size
            X_padded = np.pad(X, ((0, 0), (0, pad_size)), 'constant')
        else:
            self.d = self.original_d
            X_padded = X
            
        self.dsub = self.d // self.M
        
        # Train subquantizers with batching
        for i in range(self.M):
            start_idx = i * self.dsub
            end_idx = start_idx + self.dsub
            
            # Process in batches to reduce memory
            kmeans = MiniBatchKMeans(n_clusters=actual_k, batch_size=self.batch_size)
            for j in range(0, len(X_padded), self.batch_size):
                batch = X_padded[j:j+self.batch_size, start_idx:end_idx]
                kmeans.partial_fit(batch)
                
            self.codebooks.append(kmeans.cluster_centers_)

    def encode(self, X: np.ndarray) -> np.ndarray:
        # Pad if needed
        if self.original_d != self.d:
            X = np.pad(X, ((0, 0), (0, self.d - self.original_d)), 'constant')
        
        codes = np.empty((X.shape[0], self.M), dtype=np.uint8)
        
        # Process in batches
        for i in range(0, len(X), self.batch_size):
            batch = X[i:i+self.batch_size]
            for j in range(self.M):
                start_idx = j * self.dsub
                end_idx = start_idx + self.dsub
                subvectors = batch[:, start_idx:end_idx]
                
                # Memory efficient distance computation
                dist = np.zeros((len(subvectors), len(self.codebooks[j])))
                for k in range(len(self.codebooks[j])):
                    dist[:, k] = np.sum((subvectors - self.codebooks[j][k]) ** 2, axis=1)
                    
                codes[i:i+len(batch), j] = np.argmin(dist, axis=1)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        reconstructed = np.empty((codes.shape[0], self.d))
        
        # Process in batches
        for i in range(0, len(codes), self.batch_size):
            batch_codes = codes[i:i+self.batch_size]
            for j in range(self.M):
                start_idx = j * self.dsub
                end_idx = start_idx + self.dsub
                reconstructed[i:i+len(batch_codes), start_idx:end_idx] = self.codebooks[j][batch_codes[:, j]]
        
        return reconstructed[:, :self.original_d]