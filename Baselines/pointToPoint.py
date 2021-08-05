# %%
import os 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LassoCV
import numpy as np 
import matplotlib.pyplot as plt
import xgboost as xgb
# %%
'''
getvalue:
    path:csv文件路径
    cols:需要返回的columns的名字
evaluate_metrics
    y_true: (n,)
    y_pred: (n,)
    scaler
'''
def getvalue(path,cols):
    tmp = pd.read_csv(path)
    return tmp[cols].values
def evaluate_metrics(y_true,y_pred,scaler):
    y_pred = scaler.inverse_transform(y_pred.reshape(-1,1)).reshape(-1)
    y_true = scaler.inverse_transform(y_true.reshape(-1,1)).reshape(-1)
    d = np.abs(y_pred-y_true)
    mae = round(d.mean(),3)
    rmse = round(np.sqrt((d**2).mean()),3)

    mask = (y_true!=0)
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    d = np.abs(y_pred-y_true)
    mape = round((d / np.abs(y_true)).mean()*100,3)
    smape = round((2*d/(np.abs(y_pred)+np.abs(y_true))).mean()*100,3)
    
    # stot = ((y_pred-y_pred.mean())**2).mean()
    # r2 = 1 - rmse**2/stot
    # r2 = round(r2,3)
    #print("mae:%s mape:%s rmse:%s smape:%s r2:%s"%(mae,mape,rmse,smape,r2))
    print("mae\tmape\trmse\tsmape\tr2")
    print(mae,mape,rmse,smape)
# %% LR & MLR & XGBoost
def regression(method,city,target):
    cities = ['tangshan','hengshui']
    if method=='LR':
        features = {'tangshan':target,'hengshui':target}
    else:
        features = { 'tangshan':['CO','NO2','SO2','O3','PM25','PM10','TEMPERATURE','HUMIDITY','PM05N','PM1N','PM25N','PM10N','PM_ZUFEN','PMN_SUM','PM10N_ratio','PM25N_ratio'],
            'hengshui':['CO','NO2','SO2','O3','PM25','PM10','TEMPERATURE','HUMIDITY']}
    print(method,target)
    xfiles = {'tangshan':['769.csv','801.csv','842.csv'],'hengshui':['741.csv','910.csv','975.csv','995.csv']}
    
    print("============%s========="%(city))
    y = getvalue('../datasets/'+city+'/static.csv',target)
    scaler_y = MinMaxScaler()
    len_train = int(0.9*len(y))
    train_y,test_y = y[:len_train],y[len_train:]
    train_y = scaler_y.fit_transform(train_y.reshape(-1,1)).reshape(-1)
    test_y = scaler_y.transform(test_y.reshape(-1,1)).reshape(-1)
    
    for f in xfiles[city]:
        ########################
        # if city=='hengshui'and f!='910.csv':
        #     continue
        # if city=='tangshan' and f!='801.csv':
        #     continue
        ########################
        x = getvalue('../datasets/'+city+'/'+f,features[city])
        scaler_x = MinMaxScaler()
        train_x,test_x = x[:len_train],x[len_train:]
        if method == 'LR':
            train_x,test_x = train_x.reshape(-1,1),test_x.reshape(-1,1)
        train_x = scaler_x.fit_transform(train_x)
        test_x = scaler_x.transform(test_x)
        if method=='XGB':
            model = xgb.XGBRegressor()
        elif method=='SVR':
            model = SVR(epsilon=0.05)
        elif method=='LASSO':
            model = LassoCV()
        else:
            model = LinearRegression()
        model.fit(train_x,train_y)
        y_pred = model.predict(test_x)
        print(f[-7:-4])#输出微测站编号
        #########################
        # y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).reshape(-1)
        # y_true = scaler_y.inverse_transform(test_y.reshape(-1,1)).reshape(-1)
        # return y_pred, y_true
        ########################
        evaluate_metrics(test_y,y_pred,scaler_y)
# x,y = regression('LASSO','hengshui','CO') 
regression('SVR','tangshan','CO') 
# %% MLS-RF
# k:the number of features used in the model,
def AIC(y_true, y_pred, k,scaler_y):
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).reshape(-1)
    y_true = scaler_y.inverse_transform(y_true.reshape(-1,1)).reshape(-1)
    resid = y_true - y_pred
    l = len(y_true)
    SSR = sum(resid ** 2)
    AICValue = 2*k+l*np.log(float(SSR)/l)
    return AICValue

def twopharse(x,y):
    AICmin = np.inf 
    fset = set()
    n = x.shape[1]
    len_train = int(0.9*len(y))
    train_x,train_y = x[:len_train],y[:len_train]
    test_x,test_y = x[len_train:],y[len_train:]
    scaler_x,scaler_y = MinMaxScaler(),MinMaxScaler()
    train_x = scaler_x.fit_transform(train_x)
    test_x = scaler_x.transform(test_x)
    train_y = scaler_y.fit_transform(train_y.reshape(-1,1)).reshape(-1)
    test_y = scaler_y.transform(test_y.reshape(-1,1)).reshape(-1)
    for j in range(n):
        sub = -1
        for k in set(np.arange(n))-fset:
            mask = list(fset)+[j]
            model = LinearRegression()
            model.fit(train_x[:,mask],train_y)
            train_y_pred = model.predict(train_x[:,mask])
            t = AIC(train_y,train_y_pred,len(mask),scaler_y)
            if t < AICmin:
                AICmin = t
                sub = j
        if sub!=-1:
            fset.add(sub)
        else:
            break
    fset = list(fset)
    linear = LinearRegression()
    linear.fit(train_x[:,fset],train_y)
    train_y_pred = linear.predict(train_x[:,fset])
    reserr = train_y-train_y_pred
    nonlinear = RandomForestRegressor(n_estimators=60)
    nonlinear.fit(train_x[:,fset],reserr)
    
    # begin test
    test_y_pred = linear.predict(test_x[:,fset])+nonlinear.predict(test_x[:,fset])
    evaluate_metrics(test_y,test_y_pred,scaler_y)
    
# features = { 'tangshan':['CO','NO2','SO2','O3','PM25','PM10','TEMPERATURE','HUMIDITY','PM05N','PM1N','PM25N','PM10N','PM_ZUFEN','PMN_SUM','PM10N_ratio','PM25N_ratio','hour','dow'],
#             'hengshui':['CO','NO2','SO2','O3','PM25','PM10','TEMPERATURE','HUMIDITY','hour','dow']}
# city = 'tangshan'

# for target in ['CO','O3']:
#     print('======%s,%s======='%(city,target))
#     path = '../datasets/'+city+'/'
#     xfiles = os.listdir(path)
#     xfiles.remove('static.csv')
#     y = getvalue(path+'static.csv',target)
#     for f in xfiles:
#         print(f)
#         x = getvalue(path+f,features[city])
#         twopharse(x,y)
# %%
