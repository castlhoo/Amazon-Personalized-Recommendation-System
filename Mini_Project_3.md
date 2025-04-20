```python
pip install scikit-surprise
```

    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: scikit-surprise in c:\users\ksung\appdata\roaming\python\python312\site-packages (1.1.4)
    Requirement already satisfied: joblib>=1.2.0 in c:\programdata\anaconda3\lib\site-packages (from scikit-surprise) (1.4.2)
    Requirement already satisfied: numpy>=1.19.5 in c:\programdata\anaconda3\lib\site-packages (from scikit-surprise) (1.26.4)
    Requirement already satisfied: scipy>=1.6.0 in c:\programdata\anaconda3\lib\site-packages (from scikit-surprise) (1.13.1)
    Note: you may need to restart the kernel to use updated packages.
    


```python
import surprise 

print(surprise.__version__)
```

    1.1.4
    

1. Import Data


```python
import os
import pandas as pd
csv_path = './amazon.csv'

amazon_df = pd.read_csv(csv_path)
print(amazon_df.shape)
```

    (1465, 16)
    

2. EDA


```python
amazon_df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_id</th>
      <th>product_name</th>
      <th>category</th>
      <th>discounted_price</th>
      <th>actual_price</th>
      <th>discount_percentage</th>
      <th>rating</th>
      <th>rating_count</th>
      <th>about_product</th>
      <th>user_id</th>
      <th>user_name</th>
      <th>review_id</th>
      <th>review_title</th>
      <th>review_content</th>
      <th>img_link</th>
      <th>product_link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B07JW9H4J1</td>
      <td>Wayona Nylon Braided USB to Lightning Fast Cha...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>₹399</td>
      <td>₹1,099</td>
      <td>64%</td>
      <td>4.2</td>
      <td>24,269</td>
      <td>High Compatibility : Compatible With iPhone 12...</td>
      <td>AG3D6O4STAQKAY2UVGEUV46KN35Q,AHMY5CWJMMK5BJRBB...</td>
      <td>Manav,Adarsh gupta,Sundeep,S.Sayeed Ahmed,jasp...</td>
      <td>R3HXWT0LRP0NMF,R2AJM3LFTLZHFO,R6AQJGUP6P86,R1K...</td>
      <td>Satisfied,Charging is really fast,Value for mo...</td>
      <td>Looks durable Charging is fine tooNo complains...</td>
      <td>https://m.media-amazon.com/images/W/WEBP_40237...</td>
      <td>https://www.amazon.in/Wayona-Braided-WN3LG1-Sy...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B098NS6PVG</td>
      <td>Ambrane Unbreakable 60W / 3A Fast Charging 1.5...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>₹199</td>
      <td>₹349</td>
      <td>43%</td>
      <td>4.0</td>
      <td>43,994</td>
      <td>Compatible with all Type C enabled devices, be...</td>
      <td>AECPFYFQVRUWC3KGNLJIOREFP5LQ,AGYYVPDD7YG7FYNBX...</td>
      <td>ArdKn,Nirbhay kumar,Sagar Viswanathan,Asp,Plac...</td>
      <td>RGIQEG07R9HS2,R1SMWZQ86XIN8U,R2J3Y1WL29GWDE,RY...</td>
      <td>A Good Braided Cable for Your Type C Device,Go...</td>
      <td>I ordered this cable to connect my phone to An...</td>
      <td>https://m.media-amazon.com/images/W/WEBP_40237...</td>
      <td>https://www.amazon.in/Ambrane-Unbreakable-Char...</td>
    </tr>
  </tbody>
</table>
</div>




```python
amazon_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1465 entries, 0 to 1464
    Data columns (total 16 columns):
     #   Column               Non-Null Count  Dtype 
    ---  ------               --------------  ----- 
     0   product_id           1465 non-null   object
     1   product_name         1465 non-null   object
     2   category             1465 non-null   object
     3   discounted_price     1465 non-null   object
     4   actual_price         1465 non-null   object
     5   discount_percentage  1465 non-null   object
     6   rating               1465 non-null   object
     7   rating_count         1463 non-null   object
     8   about_product        1465 non-null   object
     9   user_id              1465 non-null   object
     10  user_name            1465 non-null   object
     11  review_id            1465 non-null   object
     12  review_title         1465 non-null   object
     13  review_content       1465 non-null   object
     14  img_link             1465 non-null   object
     15  product_link         1465 non-null   object
    dtypes: object(16)
    memory usage: 183.3+ KB
    


```python
amazon_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_id</th>
      <th>product_name</th>
      <th>category</th>
      <th>discounted_price</th>
      <th>actual_price</th>
      <th>discount_percentage</th>
      <th>rating</th>
      <th>rating_count</th>
      <th>about_product</th>
      <th>user_id</th>
      <th>user_name</th>
      <th>review_id</th>
      <th>review_title</th>
      <th>review_content</th>
      <th>img_link</th>
      <th>product_link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1465</td>
      <td>1465</td>
      <td>1465</td>
      <td>1465</td>
      <td>1465</td>
      <td>1465</td>
      <td>1465</td>
      <td>1463</td>
      <td>1465</td>
      <td>1465</td>
      <td>1465</td>
      <td>1465</td>
      <td>1465</td>
      <td>1465</td>
      <td>1465</td>
      <td>1465</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>1351</td>
      <td>1337</td>
      <td>211</td>
      <td>550</td>
      <td>449</td>
      <td>92</td>
      <td>28</td>
      <td>1143</td>
      <td>1293</td>
      <td>1194</td>
      <td>1194</td>
      <td>1194</td>
      <td>1194</td>
      <td>1212</td>
      <td>1412</td>
      <td>1465</td>
    </tr>
    <tr>
      <th>top</th>
      <td>B07JW9H4J1</td>
      <td>Fire-Boltt Ninja Call Pro Plus 1.83" Smart Wat...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>₹199</td>
      <td>₹999</td>
      <td>50%</td>
      <td>4.1</td>
      <td>9,378</td>
      <td>[CHARGE &amp; SYNC FUNCTION]- This cable comes wit...</td>
      <td>AHIKJUDTVJ4T6DV6IUGFYZ5LXMPA,AE55KTFVNXYFD5FPY...</td>
      <td>$@|\|TO$|-|,Sethu madhav,Akash Thakur,Burger P...</td>
      <td>R3F4T5TRYPTMIG,R3DQIEC603E7AY,R1O4Z15FD40PV5,R...</td>
      <td>Worked on iPhone 7 and didn’t work on XR,Good ...</td>
      <td>I am not big on camera usage, personally. I wa...</td>
      <td>https://m.media-amazon.com/images/I/413sCRKobN...</td>
      <td>https://www.amazon.in/Wayona-Braided-WN3LG1-Sy...</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>3</td>
      <td>5</td>
      <td>233</td>
      <td>53</td>
      <td>120</td>
      <td>56</td>
      <td>244</td>
      <td>9</td>
      <td>6</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>8</td>
      <td>3</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



2-1. CHECK MISS VALUE


```python
amazon_df['category'].isnull().sum() # ALL FINE
amazon_df['category'].head()
```




    0    Computers&Accessories|Accessories&Peripherals|...
    1    Computers&Accessories|Accessories&Peripherals|...
    2    Computers&Accessories|Accessories&Peripherals|...
    3    Computers&Accessories|Accessories&Peripherals|...
    4    Computers&Accessories|Accessories&Peripherals|...
    Name: category, dtype: object



2-2. Manipulate Category and userId data


```python
amazon_df['category'] = amazon_df['category'].apply(lambda x : x.split('|')[-2])
```


```python
amazon_df['category'].head()
```




    0    Cables
    1    Cables
    2    Cables
    3    Cables
    4    Cables
    Name: category, dtype: object



2-3. Extract Data need only for ML


```python
amazon_df = amazon_df[['product_name', 'product_id','category','user_id','rating','rating_count']]
```


```python
amazon_df['rating'].describe()
```




    count     1465
    unique      28
    top        4.1
    freq       244
    Name: rating, dtype: object



2-4. Changing Type of Data


```python
import numpy as np
amazon_df['rating'] = amazon_df['rating'].replace('|', np.nan)
amazon_df['rating'] = amazon_df['rating'].astype(float)
```


```python
amazon_df['rating'].isnull().sum() # -> 1
```




    1




```python
amazon_df['rating'].describe() # fillna(mean)
```




    count    1464.000000
    mean        4.096585
    std         0.291674
    min         2.000000
    25%         4.000000
    50%         4.100000
    75%         4.300000
    max         5.000000
    Name: rating, dtype: float64




```python
amazon_df['rating'].fillna(amazon_df['rating'].mean(), inplace=True)
```

    C:\Users\ksung\AppData\Local\Temp\ipykernel_21780\4076523193.py:1: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
    
    
      amazon_df['rating'].fillna(amazon_df['rating'].mean(), inplace=True)
    


```python
amazon_df['rating_count'] = amazon_df['rating_count'].str.replace(",", "")
amazon_df['rating_count'].fillna(0, inplace=True)
amazon_df['rating_count'] = amazon_df['rating_count'].astype(int)
```

    C:\Users\ksung\AppData\Local\Temp\ipykernel_21780\3839545833.py:2: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
    
    
      amazon_df['rating_count'].fillna(0, inplace=True)
    


```python
print(amazon_df[amazon_df['rating_count'] == 0].value_counts()) # 2
amazon_df['rating_count'].describe() # -> median

amazon_df[amazon_df['rating_count']==0].fillna(amazon_df['rating_count'].median(), inplace=True)
```

    product_name                                                                                                                                                                                   product_id  category  user_id                       rating  rating_count
    Amazon Brand - Solimo 65W Fast Charging Braided Type C to C Data Cable | Suitable For All Supported Mobile Phones (1 Meter, Black)                                                             B0B94JPY2N  Cables    AE7CFHY23VAJT2FI4NZKKP6GS2UQ  3.0     0               1
    REDTECH USB-C to Lightning Cable 3.3FT, [Apple MFi Certified] Lightning to Type C Fast Charging Cord Compatible with iPhone 14/13/13 pro/Max/12/11/X/XS/XR/8, Supports Power Delivery - White  B0BQRJ3C47  Cables    AGJC5O5H5BBXWUV7WRIEIOOR3TVQ  5.0     0               1
    Name: count, dtype: int64
    

    C:\Users\ksung\AppData\Local\Temp\ipykernel_21780\277544766.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      amazon_df[amazon_df['rating_count']==0].fillna(amazon_df['rating_count'].median(), inplace=True)
    


```python
amazon_df.isnull().sum()
```




    product_name    0
    product_id      0
    category        0
    user_id         0
    rating          0
    rating_count    0
    dtype: int64




```python
amazon_df.shape
```




    (1465, 6)




```python
amazon_recommender_df = amazon_df.copy()
```


```python
# One userId cell has multiple userid, seperate and make dataframe
def explode_user_ids(df):
    df = df.copy()
    df['user_id'] = df['user_id'].astype(str)
    df['user_id'] = df['user_id'].str.split(',')
    return df.explode('user_id').reset_index(drop=True)


amazon_recommender_df = explode_user_ids(amazon_recommender_df)

```


```python
# Making Missing value for recommendation
import numpy as np

amazon_recommender_df.loc[amazon_recommender_df.sample(frac=0.3).index, "rating"] = np.nan
```


```python
amazon_recommender_df['user_id'].head()
```




    0    AG3D6O4STAQKAY2UVGEUV46KN35Q
    1    AHMY5CWJMMK5BJRBBSNLYT3ONILA
    2    AHCTC6ULH4XB6YHDY6PCH2R772LQ
    3    AGYHHIERNXKA6P5T7CZLXKVPT7IQ
    4    AG4OGOFWXJZTQ2HKYIOCOY3KXF2Q
    Name: user_id, dtype: object




```python
amazon_recommender_df.shape
```




    (11503, 6)



2-5. Check Correlations


```python
import plotly.express as px
corr = amazon_df.corr(numeric_only=True)
print(corr)
```

                    rating  rating_count
    rating        1.000000      0.101578
    rating_count  0.101578      1.000000
    

3. Recommedation with Surprise


```python
from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV

# Data Loading
reader = Reader(rating_scale=(2, 5))
data = Dataset.load_from_df(amazon_recommender_df[['user_id', 'product_id', 'rating']], reader)

# Data Split
trainset, testset = train_test_split(data, test_size = 0.3, random_state=0)

# SVD Algorithm
algo = SVD(random_state=0)

# HyperParameter
param_grid = {'n_epochs' : [10,20,40,50], 'n_factors' : [50,100,200]}

gs = GridSearchCV(SVD, param_grid, measures=['rmse','mae'], cv = 3)
gs.fit(data)

best_algo = gs.best_estimator['rmse']
print("Best RMSE params:", gs.best_params['rmse'])
best_algo.fit(trainset)

# Cross-validate
cross_validate(best_algo, data, measures=['RMSE','MAE'], cv = 5)
predictions = best_algo.test(testset)
```

    Best RMSE params: {'n_epochs': 10, 'n_factors': 50}
    


```python
amazon_recommender_df.head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_name</th>
      <th>product_id</th>
      <th>category</th>
      <th>user_id</th>
      <th>rating</th>
      <th>rating_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wayona Nylon Braided USB to Lightning Fast Cha...</td>
      <td>B07JW9H4J1</td>
      <td>Cables</td>
      <td>AG3D6O4STAQKAY2UVGEUV46KN35Q</td>
      <td>4.2</td>
      <td>24269</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Wayona Nylon Braided USB to Lightning Fast Cha...</td>
      <td>B07JW9H4J1</td>
      <td>Cables</td>
      <td>AHMY5CWJMMK5BJRBBSNLYT3ONILA</td>
      <td>NaN</td>
      <td>24269</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Wayona Nylon Braided USB to Lightning Fast Cha...</td>
      <td>B07JW9H4J1</td>
      <td>Cables</td>
      <td>AHCTC6ULH4XB6YHDY6PCH2R772LQ</td>
      <td>4.2</td>
      <td>24269</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Wayona Nylon Braided USB to Lightning Fast Cha...</td>
      <td>B07JW9H4J1</td>
      <td>Cables</td>
      <td>AGYHHIERNXKA6P5T7CZLXKVPT7IQ</td>
      <td>NaN</td>
      <td>24269</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Wayona Nylon Braided USB to Lightning Fast Cha...</td>
      <td>B07JW9H4J1</td>
      <td>Cables</td>
      <td>AG4OGOFWXJZTQ2HKYIOCOY3KXF2Q</td>
      <td>4.2</td>
      <td>24269</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>boAt Rugged v3 Extra Tough Unbreakable Braided...</td>
      <td>B0789LZTCJ</td>
      <td>Cables</td>
      <td>AFNWJUWJRHCC6HN52KMG5AKZY37Q</td>
      <td>4.2</td>
      <td>94363</td>
    </tr>
    <tr>
      <th>96</th>
      <td>AmazonBasics Flexible Premium HDMI Cable (Blac...</td>
      <td>B07KSMBL2H</td>
      <td>Cables</td>
      <td>AEYJ5I6JZZPOJB6MGWRQOHRQLPSQ</td>
      <td>NaN</td>
      <td>426973</td>
    </tr>
    <tr>
      <th>97</th>
      <td>AmazonBasics Flexible Premium HDMI Cable (Blac...</td>
      <td>B07KSMBL2H</td>
      <td>Cables</td>
      <td>AFY5TVFOMVHGBPBTIJODYDQRZM5Q</td>
      <td>4.4</td>
      <td>426973</td>
    </tr>
    <tr>
      <th>98</th>
      <td>AmazonBasics Flexible Premium HDMI Cable (Blac...</td>
      <td>B07KSMBL2H</td>
      <td>Cables</td>
      <td>AE3O6366WGEQAANKJ76QETTUQQTQ</td>
      <td>4.4</td>
      <td>426973</td>
    </tr>
    <tr>
      <th>99</th>
      <td>AmazonBasics Flexible Premium HDMI Cable (Blac...</td>
      <td>B07KSMBL2H</td>
      <td>Cables</td>
      <td>AEQIJCPWSBCDKUO5VROXXHWX3PPA</td>
      <td>4.4</td>
      <td>426973</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 6 columns</p>
</div>




```python
uid = 'AE3O6366WGEQAANKJ76QETTUQQTQ'
iid = 'B07KSMBL2H'

pred = best_algo.predict(uid, iid, verbose=True)
print(pred)
```

    user: AE3O6366WGEQAANKJ76QETTUQQTQ item: B07KSMBL2H r_ui = None   est = 5.00   {'was_impossible': False}
    user: AE3O6366WGEQAANKJ76QETTUQQTQ item: B07KSMBL2H r_ui = None   est = 5.00   {'was_impossible': False}
    


```python
# 학습 데이터의 rating 분포 확인
print(amazon_recommender_df['rating'].describe())
print(amazon_recommender_df['rating'].value_counts())

```

    count    8052.000000
    mean        4.097788
    std         0.284223
    min         2.000000
    25%         4.000000
    50%         4.100000
    75%         4.300000
    max         5.000000
    Name: rating, dtype: float64
    rating
    4.100000    1327
    4.200000    1272
    4.300000    1258
    4.000000    1021
    4.400000     678
    3.900000     671
    3.800000     482
    4.500000     413
    3.700000     235
    3.600000     193
    3.500000     145
    4.600000      91
    3.300000      73
    3.400000      54
    4.700000      37
    3.000000      21
    3.100000      15
    4.800000      14
    2.800000      11
    3.200000       8
    2.600000       8
    2.900000       8
    5.000000       7
    2.300000       5
    4.096585       3
    2.000000       2
    Name: count, dtype: int64
    
