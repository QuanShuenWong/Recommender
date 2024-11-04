# predictor.py
import pickle
import pandas as pd
import os

# Load the trained model
with open("model.pkl", "rb") as f:
    algo = pickle.load(f)

# Load ratings data
file_path = os.path.expanduser(
    "/Users/quanshuen/PycharmProjects/Recommender/Data/yelp_dataset/yelp_academic_dataset_testData.csv")

# Read the raw content for diagnostic purposes
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_lines = file.readlines()
        print("First few lines of the raw CSV file:")
        for line in raw_lines[:10]:  # Show the first 10 lines
            print(line.strip())
except Exception as e:
    print(f"Error reading the CSV file: {e}")

# Load ratings data with tab separator, ensuring to handle extra spaces
try:
    df = pd.read_csv(file_path, sep='\t', header=None, names=["user", "item", "rating", "timestamp"], engine='python',
                     quoting=3)
except Exception as e:
    print(f"Error reading the CSV file with tab separator: {e}")

# Check the first few rows of the DataFrame
print("First few rows of ratings data:")
print(df.head())

# Display the DataFrame shape and columns
print("DataFrame shape:", df.shape)
print("DataFrame columns:", df.columns)

# Clean the DataFrame by stripping whitespace from all entries
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Convert data types
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')  # Convert rating to numeric
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')  # Convert timestamp to datetime

# Load business name map
business_name_path = os.path.expanduser(
    "/Users/quanshuen/PycharmProjects/Recommender/Data/yelp_dataset/business_id_name_map.csv")
business_names = pd.read_csv(business_name_path).set_index("business_id")

# Get unique item IDs (restaurants)
all_items = df["item"].unique()


def get_recommendations(user_id):
    # Get rated items for the user
    rated_items = df[df["user"] == user_id]["item"].values
    rated_items_set = set(rated_items)  # Use a set for faster lookup

    # Get all unique items in the dataset
    all_items = df["item"].unique()

    # Filter to get unrated items
    unrated_items = [item for item in all_items if item not in rated_items_set]

    print(f"User {user_id} rated {len(rated_items)} items.")
    print(f"Found {len(unrated_items)} unrated items for user {user_id}. Unrated items: {unrated_items}")

    # If no unrated items are available
    if not unrated_items:
        return [("No available recommendations", None)]

    # Predict ratings for unrated items
    predictions = []
    for item in unrated_items:
        pred = algo.predict(user_id, item).est  # Predict using actual item IDs
        print(f"Predicted rating for user {user_id} and item {item}: {pred}")  # Debugging output
        predictions.append((item, pred))

    # Check for unique predictions
    unique_predictions = set(pred[1] for pred in predictions)
    print(f"Number of unique predictions: {unique_predictions}")

    # Sort by predicted rating and get the top 5
    top_5 = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]

    # Map business IDs to names and round ratings to 1 decimal place
    top_5_with_names = []
    for item, rating in top_5:
        business_name = business_names.loc[item, 'name'] if item in business_names.index else "Unknown Business"
        rounded_rating = round(rating, 1)  # Round to 1 decimal place
        top_5_with_names.append((business_name, rounded_rating))

    return top_5_with_names  # Return a list of tuples with business names and their predicted ratings

# Example usage:
# recommendations = get_recommendations("mh_-eMZ6K5RLWhZyISBhwA")
# print(recommendations)
