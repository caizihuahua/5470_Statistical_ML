%%% This Version: June 2021. @copyright Shihao Gu, Bryan Kelly and Dacheng Xiu
%%% If you use the dataset, please cite the paper "Empirical Asset Pricing via Machine Learning" (2020) and "Autoencoder Asset Pricing Models." (2020)

Firm Characteristics Dataset Description:

Columns:
permno: CRSP Permanent Company Number
DATE: The end day of each month (YYYYMMDD) 
RET: CRSP Holding Returns with dividends in that month (can used as response variable) 


94 Lagged Firm Characteristics (Details are in the appendix of our paper)
sic2: The first two digits of the Standard Industrial Classification code on DATE

Most of these characteristics are released to the public with a delay. To avoid the forward-looking bias, we assume that monthly characteristics are delayed by at most 1 month, quarterly with at least 4 months lag, and annual with at least 6 months lag. Therefore, in order to predict returns at month t + 1, we use most recent monthly characteristics at the end of month t, most recent quarterly data by end t − 4, and most recent annual data by end t − 6. In this dataset, we've already adjusted the lags. (e.g. When DATE=19570329 in our dataset, you can use the monthly RET at 195703 as the response variable.) 


