import numpy as np

def cos_sim(v1, v2):
  norm1 = np.sqrt(np.sum(np.square(v1)))
  norm2 = np.sqrt(np.sum(np.square(v2)))

  return np.dot(v1, v2)/(norm1 * norm2)