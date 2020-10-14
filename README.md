# Hous Price Prediction Model

[@CZ](zcczhang.github.io)

This is the first practice for machine learning and for Kaggle competition: [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).
Using Ridge, Lasso, LGBM, XGB, Stacking CV Regressor, and etc, to reach Score(mean absolute error): 11977.59807; ***13<sup>th</sup> place*** out of 19,465 teams ***(0.06%)*** For more information.


<br>

![](https://raw.githubusercontent.com/zcczhang/House_Price_Prediction_Model/master/output/rank.png)

<br>

[version 1:]() 
Simple Prediction(with ridge regression, random forest, bagging, XGBoost) [View PDF](https://zcczhang.github.io/files/House_Price_Prediction_v1.pdf)

[version 2:](https://github.com/zcczhang/House_Price_Prediction_Model/blob/master/house_price_prediction_v2.ipynb) Score(root mean squared logarithmic error): 0.10643; ***Rank: top 2%***. score: 0.10643; [Project Demo](https://zcczhang.github.io/projects/house_pice_prediction)

version3: Score(mean absolute error): 11977.59807; ***Rank: 13 out of 19,465 teams(0.06%)***



# **House Price Prediction --- version 2**
***Charles Zhang*** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Jan 19 2020**

# Introduction
![](https://i.ytimg.com/vi/LvfbopVq-WE/maxresdefault.jpg)

#### This is my second version ***House Price Prediction*** Model for the Kaggle Competition. In this version, I improve methods for processing missing value more accurately, use Ridge, Lasso, LGBM, XGB, and Stacking CV Regressor to build machie learning models, and add blended models. Since some contents are repeated, I will just breifly describe the dataset at the beginning.



```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.gridspec as gridspec
from datetime import datetime
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import matplotlib.style as style
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


import warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
```

# A Glimpse of the datasets.


```python
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
```


```python
# gives us statistical info about the numerical variables. 
train.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Id</td>
      <td>1460.0</td>
      <td>730.500000</td>
      <td>421.610009</td>
      <td>1.0</td>
      <td>365.75</td>
      <td>730.5</td>
      <td>1095.25</td>
      <td>1460.0</td>
    </tr>
    <tr>
      <td>MSSubClass</td>
      <td>1460.0</td>
      <td>56.897260</td>
      <td>42.300571</td>
      <td>20.0</td>
      <td>20.00</td>
      <td>50.0</td>
      <td>70.00</td>
      <td>190.0</td>
    </tr>
    <tr>
      <td>LotFrontage</td>
      <td>1201.0</td>
      <td>70.049958</td>
      <td>24.284752</td>
      <td>21.0</td>
      <td>59.00</td>
      <td>69.0</td>
      <td>80.00</td>
      <td>313.0</td>
    </tr>
    <tr>
      <td>LotArea</td>
      <td>1460.0</td>
      <td>10516.828082</td>
      <td>9981.264932</td>
      <td>1300.0</td>
      <td>7553.50</td>
      <td>9478.5</td>
      <td>11601.50</td>
      <td>215245.0</td>
    </tr>
    <tr>
      <td>OverallQual</td>
      <td>1460.0</td>
      <td>6.099315</td>
      <td>1.382997</td>
      <td>1.0</td>
      <td>5.00</td>
      <td>6.0</td>
      <td>7.00</td>
      <td>10.0</td>
    </tr>
    <tr>
      <td>OverallCond</td>
      <td>1460.0</td>
      <td>5.575342</td>
      <td>1.112799</td>
      <td>1.0</td>
      <td>5.00</td>
      <td>5.0</td>
      <td>6.00</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>YearBuilt</td>
      <td>1460.0</td>
      <td>1971.267808</td>
      <td>30.202904</td>
      <td>1872.0</td>
      <td>1954.00</td>
      <td>1973.0</td>
      <td>2000.00</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <td>YearRemodAdd</td>
      <td>1460.0</td>
      <td>1984.865753</td>
      <td>20.645407</td>
      <td>1950.0</td>
      <td>1967.00</td>
      <td>1994.0</td>
      <td>2004.00</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <td>MasVnrArea</td>
      <td>1452.0</td>
      <td>103.685262</td>
      <td>181.066207</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>166.00</td>
      <td>1600.0</td>
    </tr>
    <tr>
      <td>BsmtFinSF1</td>
      <td>1460.0</td>
      <td>443.639726</td>
      <td>456.098091</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>383.5</td>
      <td>712.25</td>
      <td>5644.0</td>
    </tr>
    <tr>
      <td>BsmtFinSF2</td>
      <td>1460.0</td>
      <td>46.549315</td>
      <td>161.319273</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1474.0</td>
    </tr>
    <tr>
      <td>BsmtUnfSF</td>
      <td>1460.0</td>
      <td>567.240411</td>
      <td>441.866955</td>
      <td>0.0</td>
      <td>223.00</td>
      <td>477.5</td>
      <td>808.00</td>
      <td>2336.0</td>
    </tr>
    <tr>
      <td>TotalBsmtSF</td>
      <td>1460.0</td>
      <td>1057.429452</td>
      <td>438.705324</td>
      <td>0.0</td>
      <td>795.75</td>
      <td>991.5</td>
      <td>1298.25</td>
      <td>6110.0</td>
    </tr>
    <tr>
      <td>1stFlrSF</td>
      <td>1460.0</td>
      <td>1162.626712</td>
      <td>386.587738</td>
      <td>334.0</td>
      <td>882.00</td>
      <td>1087.0</td>
      <td>1391.25</td>
      <td>4692.0</td>
    </tr>
    <tr>
      <td>2ndFlrSF</td>
      <td>1460.0</td>
      <td>346.992466</td>
      <td>436.528436</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>728.00</td>
      <td>2065.0</td>
    </tr>
    <tr>
      <td>LowQualFinSF</td>
      <td>1460.0</td>
      <td>5.844521</td>
      <td>48.623081</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>572.0</td>
    </tr>
    <tr>
      <td>GrLivArea</td>
      <td>1460.0</td>
      <td>1515.463699</td>
      <td>525.480383</td>
      <td>334.0</td>
      <td>1129.50</td>
      <td>1464.0</td>
      <td>1776.75</td>
      <td>5642.0</td>
    </tr>
    <tr>
      <td>BsmtFullBath</td>
      <td>1460.0</td>
      <td>0.425342</td>
      <td>0.518911</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>BsmtHalfBath</td>
      <td>1460.0</td>
      <td>0.057534</td>
      <td>0.238753</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <td>FullBath</td>
      <td>1460.0</td>
      <td>1.565068</td>
      <td>0.550916</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>2.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>HalfBath</td>
      <td>1460.0</td>
      <td>0.382877</td>
      <td>0.502885</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <td>BedroomAbvGr</td>
      <td>1460.0</td>
      <td>2.866438</td>
      <td>0.815778</td>
      <td>0.0</td>
      <td>2.00</td>
      <td>3.0</td>
      <td>3.00</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>KitchenAbvGr</td>
      <td>1460.0</td>
      <td>1.046575</td>
      <td>0.220338</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>TotRmsAbvGrd</td>
      <td>1460.0</td>
      <td>6.517808</td>
      <td>1.625393</td>
      <td>2.0</td>
      <td>5.00</td>
      <td>6.0</td>
      <td>7.00</td>
      <td>14.0</td>
    </tr>
    <tr>
      <td>Fireplaces</td>
      <td>1460.0</td>
      <td>0.613014</td>
      <td>0.644666</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>GarageYrBlt</td>
      <td>1379.0</td>
      <td>1978.506164</td>
      <td>24.689725</td>
      <td>1900.0</td>
      <td>1961.00</td>
      <td>1980.0</td>
      <td>2002.00</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <td>GarageCars</td>
      <td>1460.0</td>
      <td>1.767123</td>
      <td>0.747315</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>2.00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>GarageArea</td>
      <td>1460.0</td>
      <td>472.980137</td>
      <td>213.804841</td>
      <td>0.0</td>
      <td>334.50</td>
      <td>480.0</td>
      <td>576.00</td>
      <td>1418.0</td>
    </tr>
    <tr>
      <td>WoodDeckSF</td>
      <td>1460.0</td>
      <td>94.244521</td>
      <td>125.338794</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>168.00</td>
      <td>857.0</td>
    </tr>
    <tr>
      <td>OpenPorchSF</td>
      <td>1460.0</td>
      <td>46.660274</td>
      <td>66.256028</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>25.0</td>
      <td>68.00</td>
      <td>547.0</td>
    </tr>
    <tr>
      <td>EnclosedPorch</td>
      <td>1460.0</td>
      <td>21.954110</td>
      <td>61.119149</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>552.0</td>
    </tr>
    <tr>
      <td>3SsnPorch</td>
      <td>1460.0</td>
      <td>3.409589</td>
      <td>29.317331</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>508.0</td>
    </tr>
    <tr>
      <td>ScreenPorch</td>
      <td>1460.0</td>
      <td>15.060959</td>
      <td>55.757415</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>480.0</td>
    </tr>
    <tr>
      <td>PoolArea</td>
      <td>1460.0</td>
      <td>2.758904</td>
      <td>40.177307</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>738.0</td>
    </tr>
    <tr>
      <td>MiscVal</td>
      <td>1460.0</td>
      <td>43.489041</td>
      <td>496.123024</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>15500.0</td>
    </tr>
    <tr>
      <td>MoSold</td>
      <td>1460.0</td>
      <td>6.321918</td>
      <td>2.703626</td>
      <td>1.0</td>
      <td>5.00</td>
      <td>6.0</td>
      <td>8.00</td>
      <td>12.0</td>
    </tr>
    <tr>
      <td>YrSold</td>
      <td>1460.0</td>
      <td>2007.815753</td>
      <td>1.328095</td>
      <td>2006.0</td>
      <td>2007.00</td>
      <td>2008.0</td>
      <td>2009.00</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <td>SalePrice</td>
      <td>1460.0</td>
      <td>180921.195890</td>
      <td>79442.502883</td>
      <td>34900.0</td>
      <td>129975.00</td>
      <td>163000.0</td>
      <td>214000.00</td>
      <td>755000.0</td>
    </tr>
  </tbody>
</table>
</div>



## Checking for Missing Values

### Missing Train values


```python
def missing_percentage(df):
    """This function takes a DataFrame(df) as input and returns two columns, total missing values and total missing values percentage"""
    ## the two following line may seem complicated but its actually very simple. 
    total = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending = False) != 0]
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)[round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2) != 0]
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])

missing_percentage(train)
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
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>PoolQC</td>
      <td>1453</td>
      <td>99.52</td>
    </tr>
    <tr>
      <td>MiscFeature</td>
      <td>1406</td>
      <td>96.30</td>
    </tr>
    <tr>
      <td>Alley</td>
      <td>1369</td>
      <td>93.77</td>
    </tr>
    <tr>
      <td>Fence</td>
      <td>1179</td>
      <td>80.75</td>
    </tr>
    <tr>
      <td>FireplaceQu</td>
      <td>690</td>
      <td>47.26</td>
    </tr>
    <tr>
      <td>LotFrontage</td>
      <td>259</td>
      <td>17.74</td>
    </tr>
    <tr>
      <td>GarageCond</td>
      <td>81</td>
      <td>5.55</td>
    </tr>
    <tr>
      <td>GarageType</td>
      <td>81</td>
      <td>5.55</td>
    </tr>
    <tr>
      <td>GarageYrBlt</td>
      <td>81</td>
      <td>5.55</td>
    </tr>
    <tr>
      <td>GarageFinish</td>
      <td>81</td>
      <td>5.55</td>
    </tr>
    <tr>
      <td>GarageQual</td>
      <td>81</td>
      <td>5.55</td>
    </tr>
    <tr>
      <td>BsmtExposure</td>
      <td>38</td>
      <td>2.60</td>
    </tr>
    <tr>
      <td>BsmtFinType2</td>
      <td>38</td>
      <td>2.60</td>
    </tr>
    <tr>
      <td>BsmtFinType1</td>
      <td>37</td>
      <td>2.53</td>
    </tr>
    <tr>
      <td>BsmtCond</td>
      <td>37</td>
      <td>2.53</td>
    </tr>
    <tr>
      <td>BsmtQual</td>
      <td>37</td>
      <td>2.53</td>
    </tr>
    <tr>
      <td>MasVnrArea</td>
      <td>8</td>
      <td>0.55</td>
    </tr>
    <tr>
      <td>MasVnrType</td>
      <td>8</td>
      <td>0.55</td>
    </tr>
    <tr>
      <td>Electrical</td>
      <td>1</td>
      <td>0.07</td>
    </tr>
  </tbody>
</table>
</div>



> ### Missing Test values


```python
missing_percentage(test)
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
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>PoolQC</td>
      <td>1456</td>
      <td>99.79</td>
    </tr>
    <tr>
      <td>MiscFeature</td>
      <td>1408</td>
      <td>96.50</td>
    </tr>
    <tr>
      <td>Alley</td>
      <td>1352</td>
      <td>92.67</td>
    </tr>
    <tr>
      <td>Fence</td>
      <td>1169</td>
      <td>80.12</td>
    </tr>
    <tr>
      <td>FireplaceQu</td>
      <td>730</td>
      <td>50.03</td>
    </tr>
    <tr>
      <td>LotFrontage</td>
      <td>227</td>
      <td>15.56</td>
    </tr>
    <tr>
      <td>GarageCond</td>
      <td>78</td>
      <td>5.35</td>
    </tr>
    <tr>
      <td>GarageQual</td>
      <td>78</td>
      <td>5.35</td>
    </tr>
    <tr>
      <td>GarageYrBlt</td>
      <td>78</td>
      <td>5.35</td>
    </tr>
    <tr>
      <td>GarageFinish</td>
      <td>78</td>
      <td>5.35</td>
    </tr>
    <tr>
      <td>GarageType</td>
      <td>76</td>
      <td>5.21</td>
    </tr>
    <tr>
      <td>BsmtCond</td>
      <td>45</td>
      <td>3.08</td>
    </tr>
    <tr>
      <td>BsmtQual</td>
      <td>44</td>
      <td>3.02</td>
    </tr>
    <tr>
      <td>BsmtExposure</td>
      <td>44</td>
      <td>3.02</td>
    </tr>
    <tr>
      <td>BsmtFinType1</td>
      <td>42</td>
      <td>2.88</td>
    </tr>
    <tr>
      <td>BsmtFinType2</td>
      <td>42</td>
      <td>2.88</td>
    </tr>
    <tr>
      <td>MasVnrType</td>
      <td>16</td>
      <td>1.10</td>
    </tr>
    <tr>
      <td>MasVnrArea</td>
      <td>15</td>
      <td>1.03</td>
    </tr>
    <tr>
      <td>MSZoning</td>
      <td>4</td>
      <td>0.27</td>
    </tr>
    <tr>
      <td>BsmtHalfBath</td>
      <td>2</td>
      <td>0.14</td>
    </tr>
    <tr>
      <td>Utilities</td>
      <td>2</td>
      <td>0.14</td>
    </tr>
    <tr>
      <td>Functional</td>
      <td>2</td>
      <td>0.14</td>
    </tr>
    <tr>
      <td>BsmtFullBath</td>
      <td>2</td>
      <td>0.14</td>
    </tr>
    <tr>
      <td>BsmtFinSF2</td>
      <td>1</td>
      <td>0.07</td>
    </tr>
    <tr>
      <td>BsmtFinSF1</td>
      <td>1</td>
      <td>0.07</td>
    </tr>
    <tr>
      <td>Exterior2nd</td>
      <td>1</td>
      <td>0.07</td>
    </tr>
    <tr>
      <td>BsmtUnfSF</td>
      <td>1</td>
      <td>0.07</td>
    </tr>
    <tr>
      <td>TotalBsmtSF</td>
      <td>1</td>
      <td>0.07</td>
    </tr>
    <tr>
      <td>SaleType</td>
      <td>1</td>
      <td>0.07</td>
    </tr>
    <tr>
      <td>Exterior1st</td>
      <td>1</td>
      <td>0.07</td>
    </tr>
    <tr>
      <td>KitchenQual</td>
      <td>1</td>
      <td>0.07</td>
    </tr>
    <tr>
      <td>GarageArea</td>
      <td>1</td>
      <td>0.07</td>
    </tr>
    <tr>
      <td>GarageCars</td>
      <td>1</td>
      <td>0.07</td>
    </tr>
  </tbody>
</table>
</div>



# Observation
* There are multiple types of features. 
* Some features have missing values. 
* Most of the features are object( includes string values in the variable).

#### Similarly, I will normalize the distrbution of the SalePrice by log next.


```python
def plotting_3_chart(df, feature):
    ## Importing seaborn, matplotlab and scipy modules. 
    import seaborn as sns
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from scipy import stats
    import matplotlib.style as style
    style.use('fivethirtyeight')

    ## Creating a customized chart. and giving in figsize and everything. 
    fig = plt.figure(constrained_layout=True, figsize=(15,10))
    ## creating a grid of 3 cols and 3 rows. 
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    #gs = fig3.add_gridspec(3, 3)

    ## Customizing the histogram grid. 
    ax1 = fig.add_subplot(grid[0, :2])
    ## Set the title. 
    ax1.set_title('Histogram')
    ## plot the histogram. 
    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)

    # customizing the QQ_plot. 
    ax2 = fig.add_subplot(grid[1, :2])
    ## Set the title. 
    ax2.set_title('QQ_plot')
    ## Plotting the QQ_Plot. 
    stats.probplot(df.loc[:,feature], plot = ax2)

    ## Customizing the Box Plot. 
    ax3 = fig.add_subplot(grid[:, 2])
    ## Set title. 
    ax3.set_title('Box Plot')
    ## Plotting the box plot. 
    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );
    
plotting_3_chart(train, 'SalePrice')
```


![png](https://raw.githubusercontent.com/zcczhang/House_Price_Prediction_Model/master/output/1.png)


These **three** charts above can tell us a lot about our target variable.
* Our target variable, **SalePrice** is not normally distributed.
* Our target variable is right-skewed. 
* There are multiple outliers in the variable. 


```python
#skewness and kurtosis
print("Skewness: " + str(train['SalePrice'].skew()))
print("Kurtosis: " + str(train['SalePrice'].kurt()))
```

    Skewness: 1.8828757597682129
    Kurtosis: 6.536281860064529



```python
## trainsforming target variable using numpy.log1p, 
train["SalePrice"] = np.log1p(train["SalePrice"])

## Plotting the newly transformed response variable
plotting_3_chart(train, 'SalePrice')
```


![png](https://raw.githubusercontent.com/zcczhang/House_Price_Prediction_Model/master/output/2.png)


As you can see the log transformation removes the normality of errors. This solves some of the other assumptions that we talked about above like Homoscedasticity. Let's make a comparison of the pre-transformed and post-transformed state of residual plots. 


```python
## Customizing grid for two plots. 
fig, (ax1, ax2) = plt.subplots(figsize = (20,6), ncols=2, sharey = False, sharex=False)
## doing the first scatter plot. 
sns.residplot(x = previous_train.GrLivArea, y = previous_train.SalePrice, ax = ax1)
## doing the scatter plot for GrLivArea and SalePrice. 
sns.residplot(x = train.GrLivArea, y = train.SalePrice, ax = ax2);
```

![png](https://raw.githubusercontent.com/zcczhang/House_Price_Prediction_Model/master/output/3.png)

Here, we can see that the pre-transformed chart on the left has heteroscedasticity, and the post-transformed chart on the right has almost an equal amount of variance across the zero lines.


```python
## Plot fig sizing. 
style.use('ggplot')
sns.set_style('whitegrid')
plt.subplots(figsize = (30,20))
## Plotting heatmap. 

# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(train.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


sns.heatmap(train.corr(), cmap=sns.diverging_palette(20, 220, n=200), mask = mask, annot=True, center = 0, );
## Give title. 
plt.title("Heatmap of all the Features", fontsize = 30);
```

![png](https://raw.githubusercontent.com/zcczhang/House_Price_Prediction_Model/master/output/4.png)



```python
## Dropping the "Id" from train and test set. 
# train.drop(columns=['Id'],axis=1, inplace=True)

train.drop(columns=['Id'],axis=1, inplace=True)
test.drop(columns=['Id'],axis=1, inplace=True)

## Saving the target values in "y_train". 
y = train['SalePrice'].reset_index(drop=True)



# getting a copy of train
previous_train = train.copy()
```


```python
## Combining train and test datasets together so that we can do all the work at once. 
all_data = pd.concat((train, test)).reset_index(drop = True)
## Dropping the target variable. 
all_data.drop(['SalePrice'], axis = 1, inplace = True)
```

## Dealing with Missing Values
> **Missing data in train and test data(all_data)**

> **Imputing Missing Values**


```python
## Some missing values are intentionally left blank, for example: In the Alley feature 
## there are blank values meaning that there are no alley's in that specific house. 
missing_val_col = ["Alley", 
                   "PoolQC", 
                   "MiscFeature",
                   "Fence",
                   "FireplaceQu",
                   "GarageType",
                   "GarageFinish",
                   "GarageQual",
                   "GarageCond",
                   'BsmtQual',
                   'BsmtCond',
                   'BsmtExposure',
                   'BsmtFinType1',
                   'BsmtFinType2',
                   'MasVnrType']

for i in missing_val_col:
    all_data[i] = all_data[i].fillna('None')
```


```python
## These features are continous variable, we used "0" to replace the null values. 
missing_val_col2 = ['BsmtFinSF1',
                    'BsmtFinSF2',
                    'BsmtUnfSF',
                    'TotalBsmtSF',
                    'BsmtFullBath', 
                    'BsmtHalfBath', 
                    'GarageYrBlt',
                    'GarageArea',
                    'GarageCars',
                    'MasVnrArea']

for i in missing_val_col2:
    all_data[i] = all_data[i].fillna(0)
    
## Replaced all missing values in LotFrontage by imputing the median value of each neighborhood. 
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform( lambda x: x.fillna(x.mean()))
```


```python
## the "OverallCond" and "OverallQual" of the house. 
# all_data['OverallCond'] = all_data['OverallCond'].astype(str) 
# all_data['OverallQual'] = all_data['OverallQual'].astype(str)

## Zoning class are given in numerical; therefore converted to categorical variables. 
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
all_data['MSZoning'] = all_data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

## Important years and months that should be categorical variables not numerical. 
# all_data['YearBuilt'] = all_data['YearBuilt'].astype(str)
# all_data['YearRemodAdd'] = all_data['YearRemodAdd'].astype(str)
# all_data['GarageYrBlt'] = all_data['GarageYrBlt'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str) 
```


```python
all_data['Functional'] = all_data['Functional'].fillna('Typ') 
all_data['Utilities'] = all_data['Utilities'].fillna('AllPub') 
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0]) 
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna("TA") 
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['Electrical'] = all_data['Electrical'].fillna("SBrkr") 

```


```python
missing_percentage(all_data)
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
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



So, there are no missing value left. 


```python
sns.distplot(all_data['1stFlrSF']);
```


![png](https://raw.githubusercontent.com/zcczhang/House_Price_Prediction_Model/master/output/5.png)



```python
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)

skewed_feats
```




    MiscVal          21.939672
    PoolArea         17.688664
    LotArea          13.109495
    LowQualFinSF     12.084539
    3SsnPorch        11.372080
    KitchenAbvGr      4.300550
    BsmtFinSF2        4.144503
    EnclosedPorch     4.002344
    ScreenPorch       3.945101
    BsmtHalfBath      3.929996
    MasVnrArea        2.621719
    OpenPorchSF       2.529358
    WoodDeckSF        1.844792
    1stFlrSF          1.257286
    GrLivArea         1.068750
    LotFrontage       1.058803
    BsmtFinSF1        0.980645
    BsmtUnfSF         0.919688
    2ndFlrSF          0.861556
    TotRmsAbvGrd      0.749232
    Fireplaces        0.725278
    HalfBath          0.696666
    TotalBsmtSF       0.671751
    BsmtFullBath      0.622415
    OverallCond       0.569314
    BedroomAbvGr      0.326568
    GarageArea        0.216857
    OverallQual       0.189591
    FullBath          0.165514
    GarageCars       -0.219297
    YearRemodAdd     -0.450134
    YearBuilt        -0.599194
    GarageYrBlt      -3.904632
    dtype: float64




```python
## Fixing Skewed features using boxcox transformation. 


def fixing_skewness(df):
    """
    This function takes in a dataframe and return fixed skewed dataframe
    """
    ## Import necessary modules 
    from scipy.stats import skew
    from scipy.special import boxcox1p
    from scipy.stats import boxcox_normmax
    
    ## Getting all the data that are not of "object" type. 
    numeric_feats = df.dtypes[df.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
    high_skew = skewed_feats[abs(skewed_feats) > 0.5]
    skewed_features = high_skew.index

    for feat in skewed_features:
        df[feat] = boxcox1p(df[feat], boxcox_normmax(df[feat] + 1))

fixing_skewness(all_data)
```


```python
sns.distplot(all_data['1stFlrSF']);
```


![png](https://raw.githubusercontent.com/zcczhang/House_Price_Prediction_Model/master/output/6.png)



```python
all_data = all_data.drop(['Utilities', 'Street', 'PoolQC',], axis=1)

# feture engineering a new feature "TotalFS"
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['YrBltAndRemod']=all_data['YearBuilt']+all_data['YearRemodAdd']

all_data['Total_sqr_footage'] = (all_data['BsmtFinSF1'] + all_data['BsmtFinSF2'] +
                                 all_data['1stFlrSF'] + all_data['2ndFlrSF'])

all_data['Total_Bathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) +
                               all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))

all_data['Total_porch_sf'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] +
                              all_data['EnclosedPorch'] + all_data['ScreenPorch'] +
                              all_data['WoodDeckSF'])

```


```python
all_data['haspool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
```


```python
all_data.shape
```




    (2917, 86)



## Creating Dummy Variables. 



```python
## Creating dummy variable 
final_features = pd.get_dummies(all_data).reset_index(drop=True)
final_features.shape
```




    (2917, 333)




```python
X = final_features.iloc[:len(y), :]

X_sub = final_features.iloc[len(y):, :]
```


```python
outliers = [30, 88, 462, 631, 1322]
X = X.drop(X.index[outliers])
y = y.drop(y.index[outliers])
```


```python
def overfit_reducer(df):
    """
    This function takes in a dataframe and returns a list of features that are overfitted.
    """
    overfit = []
    for i in df.columns:
        counts = df[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(df) * 100 > 99.94:
            overfit.append(i)
    overfit = list(overfit)
    return overfit


overfitted_features = overfit_reducer(X)

X = X.drop(overfitted_features, axis=1)
X_sub = X_sub.drop(overfitted_features, axis=1)
```


```python
X.shape,y.shape, X_sub.shape
```




    ((1453, 332), (1453,), (1459, 332))



# Fitting model(simple approach)

## Train_test split


```python
## Train test s
from sklearn.model_selection import train_test_split
## Train test split follows this distinguished code pattern and helps creating train and test set to build machine learning. 
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = .33, random_state = 0)
```


```python
X_train.shape, y_train.shape, X_test.shape, y_test.shape
```




    ((973, 332), (973,), (480, 332), (480,))



### Regularization Models
What makes regression model more effective is its ability of *regularizing*. The term "regularizing" stands for models ability **to structurally prevent overfitting by imposing a penalty on the coefficients.** 


There are three types of regularizations. 
* **Ridge**
* **Lasso**
* **Elastic Net**


    
### Ridge:
Ridge regression adds penalty equivalent to the square of the magnitude of the coefficients. This penalty is added to the least square loss function above and looks like this...


```python
## Importing Ridge. 
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
## Assiging different sets of alpha values to explore which can be the best fit for the model. 
alpha_ridge = [-3,-2,-1,1e-15, 1e-10, 1e-8,1e-5,1e-4, 1e-3,1e-2,0.5,1,1.5, 2,3,4, 5, 10, 20, 30, 40]
temp_rss = {}
temp_mse = {}
for i in alpha_ridge:
    ## Assigin each model. 
    ridge = Ridge(alpha= i, normalize=True)
    ## fit the model. 
    ridge.fit(X_train, y_train)
    ## Predicting the target value based on "Test_x"
    y_pred = ridge.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rss = sum((y_pred-y_test)**2)
    temp_mse[i] = mse
    temp_rss[i] = rss
```


```python
for key, value in sorted(temp_mse.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))
```

    0.01: 0.012058094362558029
    0.001: 0.012361806597259932
    0.5: 0.012398339976451882
    0.0001: 0.01245484128284161
    1e-05: 0.012608710731947215
    1e-15: 0.0126876442980313
    1e-08: 0.012689390127616456
    1e-10: 0.012689491191917741
    1: 0.013828461568989092
    1.5: 0.015292912807173023
    2: 0.016759826630923253
    3: 0.019679216533917247
    4: 0.02256515576039871
    5: 0.02540603527247574
    10: 0.03869750099582716
    20: 0.06016951688745736
    30: 0.07597213357728104
    40: 0.08783870545120151
    -1: 22.58422122267688
    -3: 37.77842304072701
    -2: 1127.9896486631667



```python
for key, value in sorted(temp_rss.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))
```

    0.01: 5.787885294027853
    0.001: 5.9336671666847645
    0.5: 5.951203188696909
    0.0001: 5.978323815763974
    1e-05: 6.052181151334658
    1e-15: 6.090069263055022
    1e-08: 6.090907261255897
    1e-10: 6.09095577212052
    1: 6.637661553114765
    1.5: 7.34059814744305
    2: 8.044716782843166
    3: 9.446023936280277
    4: 10.831274764991383
    5: 12.194896930788355
    10: 18.574800477997016
    20: 28.881368105979536
    30: 36.46662411709491
    40: 42.16257861657673
    -1: 10840.426186884908
    -3: 18133.64305954895
    -2: 541435.0313583203


### Lasso:
Lasso adds penalty equivalent to the absolute value of the sum of coefficients. This penalty is added to the least square loss function and replaces the squared sum of coefficients from Ridge. 



```python
from sklearn.linear_model import Lasso 
temp_rss = {}
temp_mse = {}
for i in alpha_ridge:
    ## Assigin each model. 
    lasso_reg = Lasso(alpha= i, normalize=True)
    ## fit the model. 
    lasso_reg.fit(X_train, y_train)
    ## Predicting the target value based on "Test_x"
    y_pred = lasso_reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rss = sum((y_pred-y_test)**2)
    temp_mse[i] = mse
    temp_rss[i] = rss
```


```python
for key, value in sorted(temp_mse.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))
```

    0.0001: 0.010061658258835855
    1e-05: 0.011553103092552693
    1e-08: 0.012464777509974378
    1e-10: 0.012469892082710245
    1e-15: 0.01246993993780167
    0.001: 0.01834391027644981
    0.01: 0.15998234085337285
    0.5: 0.16529633945001213
    1: 0.16529633945001213
    1.5: 0.16529633945001213
    2: 0.16529633945001213
    3: 0.16529633945001213
    4: 0.16529633945001213
    5: 0.16529633945001213
    10: 0.16529633945001213
    20: 0.16529633945001213
    30: 0.16529633945001213
    40: 0.16529633945001213
    -1: 14648689598.250006
    -2: 58594759730.8125
    -3: 131838210397.70003



```python
for key, value in sorted(temp_rss.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))
```

    0.0001: 4.82959596424121
    1e-05: 5.545489484425293
    1e-08: 5.9830932047877035
    1e-10: 5.985548199700918
    1e-15: 5.9855711701448
    0.001: 8.805076932695897
    0.01: 76.79152360961895
    0.5: 79.34224293600582
    1: 79.34224293600582
    1.5: 79.34224293600582
    2: 79.34224293600582
    3: 79.34224293600582
    4: 79.34224293600582
    5: 79.34224293600582
    10: 79.34224293600582
    20: 79.34224293600582
    30: 79.34224293600582
    40: 79.34224293600582
    -1: 7031371007160.002
    -2: 28125484670789.992
    -3: 63282340990896.01


### Elastic Net: 
Elastic Net is the combination of both Ridge and Lasso. It adds both the sum of squared coefficients and the absolute sum of the coefficients with the ordinary least square function. Let's look at the function. 



```python
from sklearn.linear_model import ElasticNet
temp_rss = {}
temp_mse = {}
for i in alpha_ridge:
    ## Assigin each model. 
    lasso_reg = ElasticNet(alpha= i, normalize=True)
    ## fit the model. 
    lasso_reg.fit(X_train, y_train)
    ## Predicting the target value based on "Test_x"
    y_pred = lasso_reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rss = sum((y_pred-y_test)**2)
    temp_mse[i] = mse
    temp_rss[i] = rss
```


```python
for key, value in sorted(temp_mse.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))
```

    0.0001: 0.010410247442255985
    1e-05: 0.011786774263401294
    1e-08: 0.012466548787037617
    1e-10: 0.012469905615434403
    1e-15: 0.012469939937937151
    0.001: 0.014971538718578314
    0.01: 0.10870291488354142
    0.5: 0.16529633945001213
    1: 0.16529633945001213
    1.5: 0.16529633945001213
    2: 0.16529633945001213
    3: 0.16529633945001213
    4: 0.16529633945001213
    5: 0.16529633945001213
    10: 0.16529633945001213
    20: 0.16529633945001213
    30: 0.16529633945001213
    40: 0.16529633945001213
    -3: 5.388825733568653
    -2: 5.470945111059094
    -1: 5.729175782943725



```python
for key, value in sorted(temp_rss.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))
```

    0.0001: 4.996918772282872
    1e-05: 5.65765164643262
    1e-08: 5.983943417778055
    1e-10: 5.985554695408507
    1e-15: 5.9855711702098295
    0.001: 7.186338584917596
    0.01: 52.17739914409985
    0.5: 79.34224293600582
    1: 79.34224293600582
    1.5: 79.34224293600582
    2: 79.34224293600582
    3: 79.34224293600582
    4: 79.34224293600582
    5: 79.34224293600582
    10: 79.34224293600582
    20: 79.34224293600582
    30: 79.34224293600582
    40: 79.34224293600582
    -3: 2586.6363521129515
    -2: 2626.053653308364
    -1: 2750.0043758129887


# Fitting model (Advanced approach)


```python
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))
    return (rmse)
```


```python
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
```


```python
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))                                
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))
```


```python
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)                             
```


```python
lightgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       learning_rate=0.01, 
                                       n_estimators=5000,
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                       verbose=-1,
                                       )
```


```python
xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)
```


```python
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, xgboost, lightgbm),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)
```


```python
# score = cv_rmse(stack_gen)
```


```python
score = cv_rmse(ridge)
print("Ridge: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(lasso)
print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(elasticnet)
print("elastic net: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(svr)
print("SVR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(lightgbm)
print("lightgbm: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

# score = cv_rmse(gbr)
# print("gbr: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(xgboost)
print("xgboost: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )
```

    Ridge: 0.1011 (0.0141)
     2020-01-22 15:12:38.941969
    LASSO: 0.0997 (0.0142)
     2020-01-22 15:12:45.519931
    elastic net: 0.0998 (0.0143)
     2020-01-22 15:13:12.882048
    SVR: 0.1020 (0.0146)
     2020-01-22 15:13:26.262319
    lightgbm: 0.1054 (0.0154)
     2020-01-22 15:13:44.348901
    [15:13:44] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [15:13:58] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [15:14:12] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [15:14:28] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [15:14:42] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [15:14:55] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [15:15:09] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [15:15:25] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [15:15:39] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [15:15:53] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    xgboost: 0.1061 (0.0147)
     2020-01-22 15:16:07.581332



```python
print('START Fit')

print('stack_gen')
stack_gen_model = stack_gen.fit(np.array(X), np.array(y))

print('elasticnet')
elastic_model_full_data = elasticnet.fit(X, y)

print('Lasso')
lasso_model_full_data = lasso.fit(X, y)

print('Ridge') 
ridge_model_full_data = ridge.fit(X, y)

print('Svr')
svr_model_full_data = svr.fit(X, y)

# print('GradientBoosting')
# gbr_model_full_data = gbr.fit(X, y)

print('xgboost')
xgb_model_full_data = xgboost.fit(X, y)

print('lightgbm')
lgb_model_full_data = lightgbm.fit(X, y)
```

    START Fit
    stack_gen
    [15:17:42] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [15:17:54] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [15:18:07] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [15:18:21] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [15:18:34] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [15:18:54] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [15:19:15] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    elasticnet
    Lasso
    Ridge
    Svr
    xgboost
    [15:19:40] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    lightgbm


# Blending Models


```python
1.0 * elastic_model_full_data.predict(X)
```




    array([12.2252765 , 12.19482971, 12.28743582, ..., 12.45057568,
           11.846052  , 11.9162269 ])




```python
def blend_models_predict(X):
    return ((0.1 * elastic_model_full_data.predict(X)) + \
            (0.05 * lasso_model_full_data.predict(X)) + \
            (0.2 * ridge_model_full_data.predict(X)) + \
            (0.1 * svr_model_full_data.predict(X)) + \
#             (0.1 * gbr_model_full_data.predict(X)) + \
            (0.15 * xgb_model_full_data.predict(X)) + \
            (0.1 * lgb_model_full_data.predict(X)) + \
            (0.3 * stack_gen_model.predict(np.array(X))))
```


```python
print('RMSLE score on train data:')
print(rmsle(y, blend_models_predict(X)))
```

    RMSLE score on train data:
    0.06279142797823006



```python
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(X_sub)))
```

    Predict submission


# Submission


```python
q1 = submission['SalePrice'].quantile(0.005)
q2 = submission['SalePrice'].quantile(0.995)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
submission.to_csv("submission.csv", index=False)
```

### Reference
Notebooks in kaggle:

[House Prices: 1st Approach to Data Science Process](https://www.kaggle.com/cheesu/house-prices-1st-approach-to-data-science-process)

[Stack&Blend LRs XGB LGB {House Prices K} v17](https://www.kaggle.com/itslek/stack-blend-lrs-xgb-lgb-house-prices-k-v17)

[EDA, New Models and Stacking](https://www.kaggle.com/datafan07/top-1-approach-eda-new-models-and-stacking)
