import numpy as np
from collections import deque

class SemanticCache():
    def __init__(self, dims, maxlen = 10, threshold = 10):
        """ Ring Buffer + Deque implementation to speed up caching. """
        self.i = 0
        self.phys_len = 0
        self.maxlen = maxlen
        self.vectors = np.zeros((maxlen, dims))
        self.dq = deque(maxlen=maxlen)
        self.threshold = threshold

    def append(self, vector, data):
        self.vectors[self.i, :] = vector
        self.i = (self.i + 1) % self.maxlen
        self.phys_len = min(self.maxlen, self.phys_len + 1)
        return self.dq.append(data)
    
    def search(self, target_vector):
        if self.phys_len > 0:
            distances = np.linalg.norm(self.vectors - target_vector, axis=-1)
            idx = np.argmin(distances)
            min_val = np.amin(distances)
            if min_val < self.threshold:
                return self.dq[idx]
        return None

if __name__=='__main__':
    # UNIT TESTING

    import numpy as np
    from semantic_cache import SemanticCache

    target_v = np.array(
        [0, 2, 3, 4, 1]
    )

    vectors = [
        [4, 5, 2, 4, 2],
        [4, 8, 9, 6, 7],
        [1, 7, 1, 5, 6],
        [1, 6, 3, 4, 4],
        [7, 0, 5, 3, 3],
        [3, 8, 5, 4, 2],
        [1, 5, 9, 1, 3],
        [4, 2, 9, 0, 0],
        [1, 7, 1, 9, 1],
        [0, 0, 8, 2, 1],
        [9, 7, 3, 4, 5],
        [6, 4, 6, 6, 3],
        [0, 2, 3, 4, 1],
        [5, 8, 7, 1, 9],
    ]

    data = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
    ]

    cache = SemanticCache(dims=5, maxlen=7)
    for v, d in zip(vectors, data):
        cache.append(np.array(v), d)

    print(cache.search(target_v))