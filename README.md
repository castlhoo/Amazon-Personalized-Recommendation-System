# Amazon Personalized Recommendation System 🚀

## 🔍 Project Overview

This project was conducted to apply what I learned about **Machine Learning-based recommendation systems** through a practical scenario. While living in Korea, I often use **Coupang**, and I always found it fascinating how, after buying a product like 'A', it recommends related products like 'A′' that I might also like. I wanted to explore how this worked under the hood.

Humans don't usually like overspending and often only want to buy what they need. But when a platform recommends the **right product at the right time**, it builds a sense of trust and friendliness towards the company. This is what led me to build a **personalized product recommender**, inspired by a real-world e-commerce system.

## 📊 Dataset

I used the **Amazon Sales Dataset** published by `KARKAVELRAJA J` on Kaggle. This dataset contains rich information about products, users, and their reviews.

🔗 [Kaggle Dataset Link](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)

### Raw Dataset

- Rows: 1465
- Columns: 16

| Column Name          | Description                       |
| -------------------- | --------------------------------- |
| product\_id          | Unique identifier for the product |
| product\_name        | Name of the product               |
| category             | Hierarchical product category     |
| discounted\_price    | Discounted price                  |
| actual\_price        | Original price                    |
| discount\_percentage | Discount rate                     |
| rating               | Product rating                    |
| rating\_count        | Number of ratings                 |
| about\_product       | Product description               |
| user\_id             | List of user IDs who reviewed     |
| user\_name           | Usernames                         |
| review\_id           | Review identifiers                |
| review\_title        | Title of the review               |
| review\_content      | Detailed review text              |
| img\_link            | Product image link                |
| product\_link        | Product purchase link             |

---

## 📊 Step 2: EDA (Exploratory Data Analysis)

### Data Preview

```python
amazon_df.head(2)
```
![image](https://github.com/user-attachments/assets/90340030-2ca5-4bc2-adb5-4c6c71a92a3d)

This shows the first two rows with multiple `user_id`, `user_name`, etc., as comma-separated lists. We later explode these values.

### Data Information

```python
amazon_df.info()
```
![image](https://github.com/user-attachments/assets/f8389317-e788-4014-ad14-05ccfac8b4e6)

Reveals all columns are non-null except for `rating_count` which has 2 missing values.

### Description of Categorical Columns

```python
amazon_df.describe()
```

- `product_id`: 1351 unique values
- `product_name`: 1337 unique names
- `category`: 211 unique category strings
- `rating`: 28 unique scores, most frequent is 4.1 (244 occurrences)
- `user_id`, `user_name`, etc., include **comma-separated values** which must be split later

---

## 🔄 Step 3: Matrix Completion

**Matrix Completion** is a technique used in recommendation systems to **fill in missing values** in the user-item matrix. Most users rate only a few products. So we assume the overall user-item matrix is low-rank and try to complete the matrix by estimating missing entries using techniques like matrix factorization.

We generate all possible user-product pairs to simulate this matrix.

### Why Matrix Completion?

In real-world recommendation systems, most of the user-item interactions are missing — that is, users only rate a small subset of products. In our dataset, we had over 12 million missing values. Matrix completion allows us to mathematically estimate the missing ratings under the assumption that user preferences and item characteristics lie in a low-dimensional space. This is fundamental in collaborative filtering and is widely used in platforms like Netflix and Amazon.

By simulating the entire user-product interaction matrix and estimating the missing values, we can recommend products that the user has not yet rated but is likely to enjoy.

### Step 1: Exploding User IDs

```python
from itertools import product

def explode_user_ids(df):
    df = df.copy()
    df['user_id'] = df['user_id'].astype(str)
    df['user_id'] = df['user_id'].str.split(',')
    return df.explode('user_id').reset_index(drop=True)

amazon_recommender_df = explode_user_ids(amazon_df)
```

### Step 2: Create (user_id, product_id) Matrix

```python
users = amazon_recommender_df['user_id'].unique()
items = amazon_recommender_df['product_id'].unique()
pairs = pd.DataFrame(product(users, items), columns=['user_id','product_id'])

cleaned_df = amazon_recommender_df.drop_duplicates(subset=['user_id', 'product_id'])

amazon_recomm_df = pairs.merge(
    cleaned_df[['user_id', 'product_id', 'rating']],
    on=['user_id', 'product_id'],
    how='left'
)

product_map = amazon_recommender_df[['product_id','product_name']].drop_duplicates()
amazon_recomm_df = amazon_recomm_df.merge(product_map, on='product_id', how='left')
```
![image](https://github.com/user-attachments/assets/fc07b92b-6c2c-4601-b9a0-42e6d642b733)

- Final Shape: **12,226,550 rows**
- Over 12 million missing ratings (we'll fill them using modeling)

---

## 🪜 Step 4: Data Preprocessing

```python
import numpy as np
amazon_recomm_df['rating'] = amazon_recomm_df['rating'].replace('|', np.nan)
amazon_recomm_df['rating'] = amazon_recomm_df['rating'].astype(float)
```

### Null Rating Count

```python
amazon_recomm_df['rating'].isnull().sum()
```

→ **12,215,954 missing values**

### Check for Duplicates

```python
amazon_recomm_df[amazon_recomm_df.duplicated()].value_counts()
```

→ No duplicated rows

### Describe Ratings

```python
amazon_recomm_df['rating'].describe()
```

```
count    10596.000000
mean         4.093196
std          0.287076
min          2.000000
25%          3.900000
50%          4.100000
75%          4.300000
max          5.000000
```

---

## 📈 Step 5: Modeling with Surprise (SVD)

```python
from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV

train_df = amazon_recomm_df[amazon_recomm_df['rating'].notnull()]
reader = Reader(rating_scale=(2, 5))
data = Dataset.load_from_df(train_df[['user_id', 'product_id', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.3, random_state=0)

param_grid = {'n_epochs': [5,10,20,30], 'n_factors': [50, 100]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse','mae'], cv=3)
gs.fit(data)

best_algo = gs.best_estimator['rmse']
best_algo.fit(trainset)
```

### Best Params

```python
print(gs.best_params['rmse'])
```

```python
{'n_epochs': 30, 'n_factors': 50}
```

---

## 🔍 Personalized Recommendation Program

```python
def recommendation():
    try:
        user_id = input("Please input your id : ").strip()
        if user_id not in amazon_recomm_df['user_id'].unique():
            print("You are not our member, Please sign in!")
            return

        unrated_products = un_rating_product(user_id)
        if not unrated_products:
            print('You have already commented in all our products. Thank you!')
            return

        result = recomm_movie(best_algo, user_id, unrated_products, top_n=10)

        print("\nTop10 what you will love!")
        for idx, (pid, name, rating) in enumerate(result, 1):
            print(f"{idx}. {name} (Expected Rating: {rating:.2f})")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print("ERROR:", e)
```

---

## 🔢 Example Results & Marketing Implications

### 👤 User: `AE3O6366WGEQAANKJ76QETTUQQTQ`

```python
Top10 what you will love!
1. Amazon Basics Wireless Mouse (Expected Rating: 4.66)
2. Sony Bravia 65" 4K Smart TV (Expected Rating: 4.57)
3. Syncwire USB Cable (Expected Rating: 4.55)
4. ESR iPad Glass Protector (Expected Rating: 4.54)
5. Belkin USB C Cable (Expected Rating: 4.53)
6. Havells Aqua Kettle (Expected Rating: 4.53)
7. Instant Pot Air Fryer (Expected Rating: 4.53)
8. Sujata Mixer Grinder (Expected Rating: 4.52)
9. WD SSD 240GB (Expected Rating: 4.48)
10. Oratech Milk Frother (Expected Rating: 4.48)
```

**Analysis**: This user is likely tech-savvy and values high-performance electronics and kitchen appliances. Marketing campaigns could promote bundled electronic products or premium gadget accessories.

### 👤 User: `AE3MSW6H3AL6F3ZGR5LCN5AHJO6A`

```python
Top10 what you will love!
1. Amazon Basics Wireless Mouse (Expected Rating: 4.56)
2. Instant Pot Air Fryer (Expected Rating: 4.49)
3. Logitech Silent Mouse (Expected Rating: 4.41)
4. Zuvexa Milk Frother (Expected Rating: 4.41)
5. Sony Bravia 65" TV (Expected Rating: 4.42)
6. Spigen Glass Protector (Expected Rating: 4.41)
7. Logitech M331 Mouse (Expected Rating: 4.40)
8. Syncwire USB Cable (Expected Rating: 4.53)
9. Oratech Frother (Expected Rating: 4.47)
10. Sujata Mixer Grinder (Expected Rating: 4.52)
```

**Analysis**: This user appears to be interested in **home & lifestyle products** with high functionality. Recommendations should focus on kitchen gadgets and ergonomic tech accessories.

---

## 📄 Final Thoughts

This project provided hands-on experience with how **recommendation engines** work under real-world constraints. I explored collaborative filtering and matrix completion using **SVD**, and successfully implemented a system that predicts missing ratings.

### 🧠 Completion Matrix Insight

Matrix Completion fills in missing values in a user-item rating matrix under the assumption that the data is low-rank. In our project, more than **12 million entries** were missing, and matrix completion allowed us to predict what ratings users would likely give to unrated products.

It mimics how Netflix, Amazon, or Coupang determine user preferences from sparse data.

---

## 👏 Conclusion

- Recommendation systems boost user retention by personalizing content.
- Even with sparse data, **SVD-based models** are powerful in predicting preferences.
- Businesses can utilize this information to **segment users**, **target promotions**, and **recommend products** effectively.

> “In the end, recommendation systems aren't just about accuracy. They're about empathy.”

---
