from surprise import Dataset, SVD, Reader, accuracy
from surprise.model_selection import train_test_split
import os
import pickle

# path to dataset file, read and load data
file_path = os.path.expanduser("/Users/quanshuen/PycharmProjects/Recommender/Data/yelp_dataset/yelp_academic_dataset_testData.csv")

reader = Reader(line_format="user item rating", sep="\t")

data = Dataset.load_from_file(file_path, reader=reader)
# training the algorithm
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

with open("model.pkl", "wb") as f:
    pickle.dump(algo, f)