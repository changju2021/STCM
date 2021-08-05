'''
DeepCM和LSTNet引用
'''
from torch.utils.data import Dataset
import pandas as pd 
import numpy as np 
import torch 
from sklearn.preprocessing import MinMaxScaler
def wgn(x, snr):
    P_signal = np.sum(abs(x)**2)/len(x)
    P_noise = P_signal/10**(snr/10.0)
    return np.random.randn(len(x)) * np.sqrt(P_noise)
def generate_noise(city,mobile,noise_num):
    df = pd.read_csv('../datasets/'+city+'/'+mobile+'.csv')
    features = { 'tangshan':['CO','NO2','SO2','O3','PM25','PM10','TEMPERATURE','HUMIDITY','PM05N','PM1N','PM25N','PM10N','PM_ZUFEN','PMN_SUM','PM10N_ratio','PM25N_ratio'],
            'hengshui':['CO','NO2','SO2','O3','PM25','PM10','TEMPERATURE','HUMIDITY']}
    data = df[features[city]].values
    mask = np.random.randint(0,data.shape[0],noise_num)
    for col in range(data.shape[1]):
        data[mask,col] = wgn(data[mask,col],10)+data[mask,col]
    new_df = pd.DataFrame(data,columns=features[city])
    new_df.to_csv('../datasets/'+city+'/'+mobile+'_'+'noise_'+str(noise_num)+'.csv',index=None)
def evaluate_metric(model, data_iter, scaler):
    model.eval()
    y_true, y_pred= [],[]
    with torch.no_grad():
        for x, y in data_iter:
            y_true += scaler.inverse_transform(y.cpu().numpy().reshape(-1,1)).reshape(-1).tolist()
            y_pred += scaler.inverse_transform(model(x).cpu().numpy().reshape(-1,1)).reshape(-1).tolist()
        y_true, y_pred = np.asarray(y_true),np.asarray((y_pred))
        
        # df = pd.DataFrame()
        # df['true']=y_true
        # df['pred']=y_pred
        # df.to_csv("tangshanO3.csv",index=None)
        
        d = np.abs(y_true - y_pred)
        MAE = round(d.mean(),3)
        RMSE = round(np.sqrt((d**2).mean()),3)
        SMAPE = round((2*d / (np.abs(y_true)+np.abs(y_pred))).mean()*100,3)
        
        mask = y_true!=0
        d = np.abs(y_true[mask]-y_pred[mask])
        MAPE = round(( d / np.abs(y_true [mask])).mean()*100,3)
        print(MAE,MAPE,RMSE,SMAPE)
def evaluate_model(model,loss,valid_iter):
    model.eval()
    l_sum,n=0.0, 0
    with torch.no_grad():
        for x,y in valid_iter:
            y_pred = model(x).view(-1)
            l = loss(y_pred,y)
            l_sum+=l.item()*len(y)
            n+=len(y)
        return l_sum/n
class MyDatasets(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
        print(x.shape)
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
    def __len__(self):
        return len(self.y)
# %%
# generate_noise("hengshui","741",8000)