import pandas as pd
import numpy as np

# Cosine similarity
def cosine_sim(x, y):
    sim = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return sim / (norm_x * norm_y)