import numpy as np
import pandas as pd

def build_graph(station_dfs, config):
    """
    Constructs an adjacency matrix A based on correlations between stations' time series.
    Nodes: station_ids. Edges connect top k neighbors.
    Returns:
      A (numpy.ndarray): Adjacency matrix (n x n)
      station_list (list): Ordered list of station IDs corresponding to rows/cols.
    """
    station_ids = list(station_dfs.keys())
    n = len(station_ids)
    
    if n <= 1:
        return np.zeros((n, n)), station_ids

    # Compute pairwise correlation on Temperature as a proxy for climate similarity
    correlations = np.zeros((n, n))
    
    for i, sid1 in enumerate(station_ids):
        for j, sid2 in enumerate(station_ids):
            if i == j:
                correlations[i, j] = -1  # Self-correlation ignored for neighbors
                continue
            
            # Align time series by index
            df1 = station_dfs[sid1]['Air temperature']
            df2 = station_dfs[sid2]['Air temperature']
            combined = pd.concat([df1, df2], axis=1).dropna()
            
            if len(combined) > 0:
                corr = combined.corr().iloc[0, 1]
                correlations[i, j] = corr if not np.isnan(corr) else 0
            else:
                correlations[i, j] = 0

    k = config['graph'].get('k_neighbors', 3)
    k = min(k, n - 1)
    
    A = np.zeros((n, n))
    for i in range(n):
        # find top k correlations
        top_k_indices = np.argsort(correlations[i, :])[-k:]
        for j in top_k_indices:
            A[i, j] = max(0, correlations[i, j])
            A[j, i] = max(0, correlations[i, j]) # undirected graph
            
    # Row normalize A
    row_sums = A.sum(axis=1, keepdims=True)
    A = np.divide(A, row_sums, out=np.zeros_like(A), where=row_sums!=0)
    
    return A, station_ids
