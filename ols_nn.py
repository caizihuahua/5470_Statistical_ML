# My imports

import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import HuberRegressor
from tqdm import tqdm
warnings.filterwarnings('ignore')
path = './GKX/'


### Data preparation
## Load 94 firm characteristics dataset
data_ch = pd.read_csv(path+'GKX_20201231.csv')
data_ch['DATE'] = pd.to_datetime(data_ch['DATE'], format='%Y%m%d') + pd.offsets.MonthEnd(0)
data_ch = data_ch[(data_ch['DATE'] >= '1957-01-31') & (data_ch['DATE'] <= '2016-12-31')]
cols = data_ch.columns.tolist()
cols_new = [x for x in cols if x not in ['permno', 'prc', 'SHROUT', 'mve0']]
data_ch = data_ch[cols_new]
#data_ch

# Detect missings of 'RET'
#print(data_ch['RET'].isnull().sum())

# Construct dummy variables
# Remove the samples with missing 'sic2'
#print(data_ch['sic2'].isnull().sum())
data_ch  = data_ch.dropna(subset=['sic2']).reset_index(drop=True)
#print(data_ch['sic2'].isnull().sum())
dummies = pd.get_dummies(data_ch['sic2'], prefix='dum_')
data_ch = data_ch.drop('sic2', axis=1)
data_ch = pd.concat([data_ch, dummies], axis=1)
#print(data_ch.shape)

# Replace all missings of firm characteristics with 0
chas = [x for x in cols_new if x not in ['DATE', 'RET', 'sic2']]
#print(chas)
#print('Total number of missing characteristics: %d' % (data_ch[chas].isnull().sum().sum()))
data_ch[chas] = data_ch[chas].fillna(0)
#print('Total number of missing characteristics: %d' % (data_ch[chas].isnull().sum().sum()))


## Load 8 macroeconomic predictors
data_ma = pd.read_csv(path+'PredictorData2023.csv')
data_ma['yyyymm'] = pd.to_datetime(data_ma['yyyymm'], format='%Y%m') + pd.offsets.MonthEnd(0)
data_ma = data_ma[(data_ma['yyyymm'] >= '1957-01-31') & (data_ma['yyyymm'] <= '2016-12-31')].reset_index(drop=True)
#data_ma

# Construct 8 macroeconomic predictors
ma_predictors = ['dp', 'ep', 'bm', 'ntis', 'tbl', 'tms', 'dfy', 'svar']
data_ma['Index'] = data_ma['Index'].str.replace(',', '').astype('float64')
data_ma['dp'] = np.log(data_ma['D12'] / data_ma['Index'])
data_ma['ep'] = np.log(data_ma['E12'] / data_ma['Index'])
data_ma.rename(columns={'b/m': 'bm'}, inplace=True)
data_ma['tms'] = data_ma['lty'] - data_ma['tbl']
data_ma['dfy'] = data_ma['BAA'] - data_ma['AAA']
data_ma = data_ma[['yyyymm'] + ma_predictors]
#data_ma

# Construct the dataset including all covariates
data_ma_long = pd.merge(data_ch['DATE'], data_ma, left_on='DATE', right_on='yyyymm', how='left').drop('yyyymm', axis=1)
for cha in chas:
    for predictor in ma_predictors:
        name = cha + '_' + predictor
        data_ch[name] = data_ch[cha] * data_ma_long[predictor]
data = data_ch
#data


## Split the dataset
def get_data_split(str, end):
    covariates = [x for x in data.columns if (x != 'RET') & (x != 'DATE')]
    X = data[(data['DATE'] >= str) & (data['DATE'] <= end)][covariates].to_numpy()
    y = data[(data['DATE'] >= str) & (data['DATE'] <= end)]['RET'].to_numpy()
    return X, y

def r2_score(y, yhat):
    r2 = 1 - sum((y - yhat) ** 2) / sum(y ** 2)
    return r2

train_str = '1957-01-31'; train_end = '1974-12-31'
val_str = '1975-01-31'; val_end = '1986-12-31'
test_str = '1987-01-31'; test_end = '2016-12-31'



### Model Fit
## OLS & OLS3 using Huber loss
ols_oos = []
ols3_oos = []
for i in tqdm(range(30)):
    # Get training dataset, test dataset split
    str = pd.to_datetime(train_str)
    end = pd.to_datetime(val_end) + pd.DateOffset(years=i)
    oos_str = end + pd.DateOffset(years=1)
    oos_end = pd.to_datetime(test_end)
    X_train, y_train = get_data_split(str, end)
    X_test, y_test = get_data_split(oos_str, oos_end)

    # Fit the OLS model using 920 features
    ols = HuberRegressor(fit_intercept=False)
    ols.fit(X_train, y_train) 
    ols_oos.append(ols.predict(X_test)) 

    # Fit OLS3 including only three covariates (size, value, and momentum)
    X_train = X_train[['mvel1', 'bm', 'mom1m']]
    X_test = X_test[['mvel1', 'bm', 'mom1m']]
    ols3 = HuberRegressor(fit_intercept=False)
    ols3.fit(X_train[['size', 'value', 'momentum']], y_train)
    ols3_oos.append(ols3.predict(X_test)) 

# Compute 30 years' out of sample R^2 for OLS and OLS3
y = get_data_split(test_str, test_end)[1]
ols_oos = np.concatenate(ols_oos)
ols3_oos = np.concatenate(ols3_oos)
ols_oos_r2 = r2_score(y, ols_oos)
ols3_oos_r2 = r2_score(y, ols3_oos)
print('30 years\' out of sample R^2 for OLS: \d' % ols_oos_r2)
print('30 years\' out of sample R^2 for OLS3: \d' % ols3_oos_r2)
with open(path+'oos_r2_output.txt', 'w') as f:
    f.write('OLS: %d' % ols_oos_r2)
    f.write('OLS3: %d' % ols3_oos_r2)





