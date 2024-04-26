#!/usr/bin/env python
# coding: utf-8

# # Define data-split scheme and R2 score
# 
#      We adopt the recursive performance evaluation scheme decribed below:
# 
#      Training sample: starts from 195703 and increases by one year' s sample in every refit.
#      Validation sample: starts from 197503 and rolls forward by 1 year in every refit, keeping the size of 12 years.      
#      Testing sample: starts from 198703 and ends in 201603.

# In[1]:


def data_split(testing_date,dt):
    end_of_test = testing_date+100
    end_of_val = testing_date
    start_of_val = testing_date-1200
    train = dt[dt["yyyymm"] < start_of_val]
    val = dt[(dt["yyyymm"] >= start_of_val)&(dt["yyyymm"] < end_of_val)]
    test = dt[(dt["yyyymm"] >=end_of_val)&(dt["yyyymm"] <= end_of_test)]
    return train,val,test


# In[2]:


def pd_ret_split(data):
    x = data.drop(['yyyymm','permno','excess_ret'],axis=1)
    y = data['excess_ret']
    return x,y


# In[3]:


def R2_score(y, pre):
    r1 = ((y-pre)**2).sum()
    r2 = (y**2).sum()
    R_square = 1-(r1/r2)
    return R_square


# In[4]:


# Convert .ipynb to .py, so that the above functions can be imported  in other files
try:  
  get_ipython().system('jupyter nbconvert --to python Data_split.ipynb')
except:
  pass

