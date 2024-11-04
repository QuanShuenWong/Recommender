import json
import pandas as pd
from collections import defaultdict

# File paths
data_path = '/Users/quanshuen/PycharmProjects/Recommender/Data/yelp_dataset/'
business_file = data_path + 'yelp_academic_dataset_business.json'
review_file = data_path + 'yelp_academic_dataset_review.json'
output_file = data_path + 'yelp_academic_dataset_testData.csv'

# Step 1: Load businesses and filter for restaurants
restaurants = set()
with open(business_file, 'r') as f:
    for line in f:
        business = json.loads(line)
        # Check that 'categories' exists and is not None, then check for "Restaurants"
        if business.get('categories') and 'Restaurants' in business['categories']:
            restaurants.add(business['business_id'])

# Step 2: Load reviews and filter
user_reviews = defaultdict(list)
with open(review_file, 'r') as f:
    for line in f:
        review = json.loads(line)

        # Filter for reviews on restaurants only
        if review['business_id'] in restaurants:
            user_reviews[review['user_id']].append({
                'user_id': review['user_id'],
                'item_id': review['business_id'],  # item_id as restaurant_id
                'rating': review['stars'],
                'timestamp': review['date']
            })

# Step 3: Filter for users with at least 20 reviews
filtered_reviews = []
for user_id, reviews in user_reviews.items():
    if len(reviews) >= 20:
        filtered_reviews.extend(reviews)

# Step 4: Limit to 100,000 reviews
filtered_reviews = filtered_reviews[:100000]

# Step 5: Create DataFrame and save to CSV with tab delimiter
df = pd.DataFrame(filtered_reviews)
df.to_csv(output_file, sep='\t', index=False, header=True)