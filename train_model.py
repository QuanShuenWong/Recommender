from surprise import Dataset, SVD, Reader
from surprise.model_selection import cross_validate
import os

# path to dataset file
file_path = os.path.expanduser("/Users/quanshuen/PycharmProjects/Recommender/Data/yelp_dataset/yelp_academic_dataset_testData.csv")

reader = Reader(line_format="user item rating", sep="\t")

# Load the movielens-100k dataset (download it if needed),
data = Dataset.load_from_file(file_path, reader=reader)

# We'll use the famous SVD algorithm.
algo = SVD()

# Run 5-fold cross-validation and print results
cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, verbose=True)