import warnings
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet
from tqdm import tqdm
import matplotlib.pyplot as plt

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


# Construct dummy variables
# Remove the samples with missing 'sic2'
data_ch  = data_ch.dropna(subset=['sic2']).reset_index(drop=True)
dummies = pd.get_dummies(data_ch['sic2'], prefix='dum_')
data_ch = data_ch.drop('sic2', axis=1)
data_ch = pd.concat([data_ch, dummies], axis=1)


# Replace all missings of firm characteristics with 0
chas = [x for x in cols_new if x not in ['DATE', 'RET', 'sic2']]
data_ch[chas] = data_ch[chas].fillna(0)

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

# Construct the dataset including all covariates
data_ma_long = pd.merge(data_ch['DATE'], data_ma, left_on='DATE', right_on='yyyymm', how='left').drop('yyyymm', axis=1)
for cha in chas:
    for predictor in ma_predictors:
        name = cha + '_' + predictor
        data_ch[name] = data_ch[cha] * data_ma_long[predictor]
data = data_ch

# concate without covariates
# data = pd.merge(data_ch, data_ma, left_on='DATE', right_on='yyyymm', how='left').drop('yyyymm', axis=1)

## Split the dataset
def get_data_split(str, end):
    # covariates = list(set(data.columns).difference({'DATE','RET'}))
    # X = data[(data['DATE'] >= str) & (data['DATE'] <= end)][covariates].values
    ch = list(set(data.columns).difference({'DATE','RET'}))
    X = data[(data['DATE'] >= str) & (data['DATE'] <= end)][ch].to_numpy()
    y = data[(data['DATE'] >= str) & (data['DATE'] <= end)]['RET'].to_numpy()
    return X, y

def r2_score(y, yhat):
    r2 = 1 - sum((y - yhat) ** 2) / sum(y ** 2)
    return r2


init_train_str = '1957-01-31'; init_train_end = '1974-12-31'
init_val_str = '1975-01-31'; init_val_end = '1986-12-31'
init_test_str = '1987-01-31'; init_test_end = '2016-12-31'
year_span = 30


### Model Fit
enet_oos_r2 = []
enet_oos_para = []
for i in range(5,year_span,1):
    # Get training dataset, test dataset split, test for one year
    str = pd.to_datetime(init_train_str)
    end = pd.to_datetime(init_val_end)+pd.DateOffset(years=i)
    oos_str = end + pd.DateOffset(years=1)
    oos_end = oos_str
    
    x_train, y_train = get_data_split(str, end)
    x_test, y_test = get_data_split(oos_str, oos_end)

    # Fit model
    enet = ElasticNet(alpha=0.01,l1_ratio=0.3) 
    enet_norm = make_pipeline(StandardScaler(), enet)
    enet_norm.fit(x_train, y_train)

    # test
    y_test_hat = enet_norm.predict(x_test).flatten()
    oos_r2= r2_score(y_test,y_test_hat)
    enet_oos_r2.append(oos_r2)
    # record
    # total_para - zero_para
    eff_para = enet.coef_.shape[0] - np.where(enet.coef_==0)[0].shape[0]
    enet_oos_para.append(eff_para)
    print("year:{0}/30, para:{1}".format(i+1,eff_para))


print('R^2 for PLS: \d' % enet_oos_r2)
enet_rec = pd.DataFrame({'r^2':enet_oos_r2,'eff_para':enet_oos_para})
enet_rec.to_csv("Enet.csv",index=False,sep=',')