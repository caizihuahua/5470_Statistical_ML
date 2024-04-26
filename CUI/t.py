import pandas as pd
 
#任意的多组列表
pls_oos_r2 = [1,2,3]
pls_para = [4,5,6]    
 
print('R^2 for PLS: \d' % pls_oos_r2)
print('K for PLS: \d' % pls_para)
pls_rec = pd.DataFrame({'r^2':pls_oos_r2,'K':pls_para})
pls_rec.to_csv("hh.csv",index=False,sep=',')