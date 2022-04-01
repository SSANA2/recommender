import enum
import numpy as np
from surprise import Dataset
import cos_similarity

data = Dataset.load_builtin('ml-100k', prompt=False)
raw_data = np.array(data.raw_ratings, dtype=int)

raw_data[:, 0] -= 1
raw_data[:, 1] -= 1

n_users = np.max(raw_data[:, 0])
n_movies = np.max(raw_data[:, 1])

shape = (n_users+1, n_movies+1)
#shape

adj_matrix = np.ndarray(shape, dtype=int)
for user_id, movie_id, rating, time in raw_data:
    adj_matrix[user_id][movie_id] = 1
#adj_matrix

my_id, my_vector = 0, adj_matrix[0]
best_match, best_match_id, best_match_vector = 9999, -1, []

for user_id, user_vector in enumerate(adj_matrix):
    if my_id != user_id:
        euclidean_dist = np.sqrt(np.sum(np.square(my_vector - user_vector)))
        if euclidean_dist < best_match:
            best_match = euclidean_dist
            best_match_id = user_id
            best_match_vector = user_vector
        # similarity = np.dot(my_vector, user_vector)
        # if similarity > best_match:
        #     best_match = similarity
        #     best_match_id = user_id
        #     best_match_vector = user_vector

print('Best_Match: {}, Best_Match_ID: {}'.format(best_match, best_match_id))

recommend_list = []
for i, log in enumerate(zip(my_vector, best_match_vector)):
    log1, log2 = log
    if log1 < 1 and log2 >0:
        recommend_list.append(i)
    
print(recommend_list)

def cos_sim(v1, v2):
  norm1 = np.sqrt(np.sum(np.square(v1)))
  norm2 = np.sqrt(np.sum(np.square(v2)))

  return np.dot(v1, v2)/(norm1 * norm2)