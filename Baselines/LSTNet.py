# %%
import sys
sys.path.append("..")
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import pandas as pd 
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
import argparse
import os
torch.manual_seed(2333)
torch.cuda.manual_seed(2333)
np.random.seed(2333)
torch.backends.cudnn.deterministic = True   
# %%
class LSTNet(nn.Module):
    def __init__(self,in_features,target_col): #target_col:target pollutant在x中的列索引
        super(LSTNet,self).__init__()
        self.in_features = in_features
        self.target_col = target_col
        self.conv = nn.Conv1d(in_features,100,kernel_size=5)
        self.gru = nn.GRU(input_size=100,hidden_size=100,batch_first=True)
        self.gruskip = nn.GRU(input_size=100,hidden_size=5)
        self.linear1 = nn.Linear(100+24*5,in_features)
        self.dropout = nn.Dropout(p=0.2)
        self.highway = nn.Linear(24,1)
    def forward(self,x):#x(128,168,16)
        batch_size = x.shape[0]
        # CNN
        c = self.conv(x.permute(0,2,1))
        c = F.relu(c)
        c = self.dropout(c) #(128, 100, 164)
        #print(c.shape)
        
        # RNN
        r = c.permute(0,2,1).contiguous()# (128, 164, 100)
        _, r = self.gru(r)
        r = self.dropout(torch.squeeze(r,0)) # (128, 100)
        #print(r.shape)
        
        # skip-RNN
        s = c[:,:,int(-6*24):].contiguous() #128,100,144
        s = s.view(batch_size,100,6,24)
        s = s.permute(2,0,3,1).contiguous()#(6,batch_size,24,100)
        s = s.view(6,batch_size*24,100)
        _,s = self.gruskip(s)
        s = s.squeeze(0)
        s = s.view(batch_size,24*5)
        s = self.dropout(s)
        
        # fuse RNN and skip-RNN
        r = torch.cat((r,s),1)
        res = self.linear1(r) # (128, 16)
        #print(res.shape)
        
        # AR
        z = x[:,-24:,:] #(128,24,16)
        z = z.permute(0,2,1).contiguous().view(-1,24)
        z = self.highway(z)
        z = z.view(-1,self.in_features)
        
        # output
        res = res + z
        res = res[:,self.target_col]
        return F.sigmoid(res)        
# %%
def data_transform(x,y,device,shuffle=True):
    seq_len = 168 #序列窗口大小
    n = y.shape[0]
    xi, yi = [],[]
    for i in range(seq_len,n+1):
        xi.append(x[i-seq_len:i]) # x[i-seq_len:i-1]
        yi.append(y[i-1])# y[i-1]
    xi, yi = np.asarray(xi),np.asarray(yi)
    xi, yi = torch.tensor(xi).float().to(device),torch.tensor(yi).to(device)
    data = MyDatasets(xi,yi)
    data_iter = torch.utils.data.DataLoader(data,batch_size=128,shuffle=shuffle) #batch_size
    return data_iter
def load_data(city,mobile,target,device):
    features = { 'tangshan':['CO','NO2','SO2','O3','PM25','PM10','TEMPERATURE','HUMIDITY','PM05N','PM1N','PM25N','PM10N','PM_ZUFEN','PMN_SUM','PM10N_ratio','PM25N_ratio'],
            'hengshui':['CO','NO2','SO2','O3','PM25','PM10','TEMPERATURE','HUMIDITY']}
    y_path = '../../datasets/'+city+'/static.csv'
    x_path = '../../datasets/'+city+'/'+mobile+'.csv'
    y = pd.read_csv(y_path)[target].values
    x = pd.read_csv(x_path)[features[city]].values
    len_train,len_valid  = int(0.8*len(y)),int(0.9*len(y))
    
    scaler_x,scaler_y = MinMaxScaler(),MinMaxScaler()
    x[:len_train] = scaler_x.fit_transform(x[:len_train])
    x[len_train:] = scaler_x.transform(x[len_train:])
    y[:len_train] = scaler_y.fit_transform(y[:len_train].reshape(-1,1)).reshape(-1)
    y[len_train:] = scaler_y.transform(y[len_train:].reshape(-1,1)).reshape(-1)
    
    train = data_transform(x[:len_train],y[:len_train],device)
    valid = data_transform(x[len_train:len_valid],y[len_train:len_valid],device)
    test = data_transform(x[len_valid:],y[len_valid:],device,False)
    
    return train,valid,test,scaler_y
# %%
def train(in_features,city,mobile,target,train=False):
    target_col = None
    if target=="CO":
        target_col = 0
    else:
        target_col = 3
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    device = torch.device("cuda"if torch.cuda.is_available()else"cpu")
    train_iter,val_iter,test_iter,scaler = load_data(city,mobile,target,device)
    model = LSTNet(in_features,target_col).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    loss = nn.L1Loss()
    model_path = 'models/'+mobile+target+'.pt' 
    print(model_path)
    min_val_loss = np.inf
    if train:
        for epoch in tqdm(range(300)):
            l_sum, n = 0.0, 0   
            model.train()
            for x, y in train_iter:
                y_pred = model(x).view(-1)
                l = loss(y_pred, y)
                optimizer.zero_grad()
                l.backward()
                nn.utils.clip_grad_norm(model.parameters(),10,2)
                optimizer.step()
                l_sum += l.item() * y.shape[0]
                n += y.shape[0]
            val_loss = evaluate_model(model, loss, val_iter)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(model.state_dict(),model_path)
            #print("epoch", epoch, ", train loss:", l_sum / n, ", validation loss:", val_loss)
    model = LSTNet(in_features,target_col).to(device)
    model.load_state_dict(torch.load(model_path))
    print(mobile)
    evaluate_metric(model,test_iter,scaler) 
# %%
def trainAll():
    num_features = {"tangshan":16,"hengshui":8}
    mobiles = {'tangshan':['769','801','842'],'hengshui':['741','910','975','995']}
    city = "tangshan"

    for target in ["O3","CO"]:
        print("======================"+target+"=====================")
        for mobile in mobiles[city]:
            train(num_features[city],city,mobile,target,False)
    
# %%
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--city",type=str,default='hengshui')
    parser.add_argument("--mobile",type=str,default='910')
    parser.add_argument("--target",type=str,default='CO')
    parser.add_argument("--noise_num",type=int,default=2000)
    args,_ = parser.parse_known_args()
    params = vars(args)
    # generate_noise(params['city'],params['mobile'],params['noise_num'])
    mobile_name = params['mobile']+'_'+'noise_'+str(params['noise_num'])
    #train(8,params['city'],mobile_name,params['target'],False)
    train(8,params['city'],mobile_name,params['target'],True)
    

# %%
