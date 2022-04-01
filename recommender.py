
from tabnanny import verbose
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate

data = Dataset.load_builtin('ml-100k', prompt=False)
#data.raw_ratings[:10]

model = SVD()
cross_validate(model, data, measures=['rmse', 'mae'], cv=5, verbose = True)
