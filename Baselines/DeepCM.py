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
torch.manual_seed(2333)
torch.cuda.manual_seed(2333)
np.random.seed(2333)
torch.backends.cudnn.deterministic = True   
# %%
class DeepCM(nn.Module):
    def __init__(self,in_features):
        super(DeepCM,self).__init__()
        self.conv = nn.Conv1d(in_features,100,kernel_size=5,stride=1,padding=2)
        self.gru = nn.GRU(input_size=100,hidden_size=50,batch_first=True)
        self.gruskip = nn.GRU(input_size=100,hidden_size=13)
        self.dec = nn.GRU(in_features,76,batch_first=True)
        self.output = nn.Linear(76,1,True)
        self.dropout = nn.Dropout(p=0.2)
    def forward(self,x):#x(128,168,16)
        c = self.conv(x.permute(0,2,1))
        c = F.relu(c)
        c = self.dropout(c) #128,100,168
       # print(c.shape)
        _,r = self.gru(c.permute(0,2,1).contiguous())
        r = self.dropout(r.squeeze()) #128,50
        #print(r.shape)
        batch_size = x.shape[0]
        s = c.view(batch_size,100,7,24)
        s = s.permute(2,0,3,1)[:,:,-2:,:].contiguous() #(7,128,2,100)
        s = s.view(7, batch_size * 2, 100)
        _, s = self.gruskip(s)
        #print(s.shape)
        s = s.view(batch_size, 2 * 13)
        s = self.dropout(s)
        r = torch.cat((r,s),1)
        #print(r.shape)
        #print(x[:,-1,:].shape)
        _,h = self.dec(x[:,-1,:].unsqueeze(1).contiguous(),r.unsqueeze(0).contiguous())
        out = self.output(h.squeeze())
        return out
        
# %%
def data_transform(x,y,device):
    seq_len = 168 #序列窗口大小
    n = y.shape[0]
    xi, yi = [],[]
    for i in range(seq_len,n+1):
        xi.append(x[i-seq_len:i]) # x[i-seq_len:i-1]
        yi.append(y[i-1])# y[i-1]
    xi, yi = np.asarray(xi),np.asarray(yi)
    xi, yi = torch.tensor(xi).float().to(device),torch.tensor(yi).to(device)
    data = MyDatasets(xi,yi)
    data_iter = torch.utils.data.DataLoader(data,batch_size=128,shuffle=True) #batch_size
    return data_iter
def load_data(city,mobile,target,device):
    features = { 'tangshan':['CO','NO2','SO2','O3','PM25','PM10','TEMPERATURE','HUMIDITY','PM05N','PM1N','PM25N','PM10N','PM_ZUFEN','PMN_SUM','PM10N_ratio','PM25N_ratio'],
            'hengshui':['CO','NO2','SO2','O3','PM25','PM10','TEMPERATURE','HUMIDITY']}
    y_path = '../datasets/'+city+'/static.csv'
    x_path = '../datasets/'+city+'/'+mobile+'.csv'
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
    test = data_transform(x[len_valid:],y[len_valid:],device)
    
    return train,valid,test,scaler_y
# %%
def train(in_features,city,mobile,target,train=False):
    device = torch.device("cuda"if torch.cuda.is_available()else"cpu")
    train_iter,val_iter,test_iter,scaler = load_data(city,mobile,target,device)
    model = DeepCM(in_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    loss = nn.L1Loss()
    model_path = 'models/'+mobile+target+'.pt' 
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
                optimizer.step()
                l_sum += l.item() * y.shape[0]
                n += y.shape[0]
            val_loss = evaluate_model(model, loss, val_iter)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(model.state_dict(),model_path)
            #print("epoch", epoch, ", train loss:", l_sum / n, ", validation loss:", val_loss)
    model = DeepCM(in_features).to(device)
    model.load_state_dict(torch.load(model_path))
    print(mobile)
    evaluate_metric(model,test_iter,scaler) 

city = "tangshan"
for target in ["O3","CO"]:
    print("======================"+target+"=====================")
    for mobile in ['769','801','842']:
        train(16,city,mobile,target,False)
    
# %%


# %%
