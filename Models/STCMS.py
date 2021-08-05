# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from load_data import *
from utils import *
import argparse
from tqdm import tqdm
import logging
import os
torch.manual_seed(2333)
torch.cuda.manual_seed(2333)
np.random.seed(2333)
torch.backends.cudnn.deterministic = True  
# %%
class STCMS(nn.Module):
    def __init__(self,hps):
        super(STCMS,self).__init__()
        self.hps = hps
        self.droput = nn.Dropout(0.2)
        self.enc1 = nn.GRU(1,self.hps['hid1'],batch_first=True)
        
        self.wgg = nn.Linear(self.hps['hid1'],self.hps['ta'])
        self.V = nn.Linear(self.hps['ta'],1)
        
        self.dec = nn.GRUCell(self.hps['n_neighbors'],self.hps['hid1'])
        self.fc = nn.Linear(self.hps['hid1'],1)
  
    def temporalAttn(self,ht,s):
        # ht (n,168,hid1) s (n,hid2)
        timestamps = list(range(self.hps['l2']-7,self.hps['l2']-1))
        timestamps += list(range(self.hps['l2']-24,-1,-24))
        ht = ht[:,timestamps,:] #n,13,hid1
        s = s.unsqueeze(1).repeat(1,len(timestamps),1) #n,13,hid2
        t = self.wgg(ht)# n,13,ga
        e = F.softmax(self.V(F.tanh(t)),dim=1) # n,13,1
        return torch.sum(ht*e,dim=1)
    
    def forward(self,x,getGA=False,getTA=False):
        # mdatax [128, 24, num_mobile, 15]
        # mdatay [128, 24, num_mobile]
        # sdatay [128, 168]
        # long-term encoder
        ht,_ = self.enc1(x[2].unsqueeze(dim=2)) # ht [n,168,hid1]
        d = self.temporalAttn(ht,ht[:,-1,:]) # n,hid1
        
        out = self.dec(x[1][:,-1,:],d)
        out = F.sigmoid(self.fc(out))
        return out #(n,1)   
        
#%%

def train(args):
    save_path = 'models/STCMS'
    model_path = save_path+'/'+args['city']+args['target']+'.pt'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda"if torch.cuda.is_available()else"cpu")
    model = STCMS(args).to(device)
    
    train_iter,val_iter,test_iter,scaler = load_data(args['city'],args['target'],device)
    loss = torch.nn.L1Loss()
    optimizer = torch.optim.RMSprop(model.parameters(),lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)
    
    if args['istrain']:
        min_val_loss = np.inf
        for epoch in tqdm(range(1, 201)):
            l_sum, n = 0.0, 0   
            model.train()
            for x, y in train_iter:
                #print(x[0].shape,x[1].shape,x[2].shape,y.shape)
                y_pred = model(x).view(-1)
                l = loss(y_pred, y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                l_sum += l.item() * y.shape[0]
                n += y.shape[0]
            scheduler.step()
            val_loss = evaluate_model(model, loss, val_iter)
            
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(model.state_dict(), model_path)
            # print("epoch", epoch, ", train loss:", l_sum / n, ", validation loss:", val_loss)
    model = STCMS(args).to(device)
    model.load_state_dict(torch.load(model_path))
    # val_loss = evaluate_model(model, loss, val_iter)
    evaluate_metric(model,test_iter,scaler)# %%

# %%
def get_params():
    parser = argparse.ArgumentParser(description='Args for STCM')
    parser.add_argument("--l1",type=int,default=24)
    parser.add_argument("--l2",type=int,default=168)
    parser.add_argument("--batch_size",type=int,default=128)
    parser.add_argument("--lr",type=float,default=0.001)
    parser.add_argument("--city",type=str,default='hengshui')
    parser.add_argument("--target",type=str,default='CO')
    parser.add_argument("--hid2",type=int,default=80)
    parser.add_argument("--hid1",type=int,default=80)
    parser.add_argument("--ga",type=int,default=120)
    parser.add_argument("--ta",type=int,default=120)
    parser.add_argument("--fuse",type=int,default=50)
    parser.add_argument("--dropout",type=float,default=0.2)
    parser.add_argument("--istrain",type=bool,default=True)
    args,_ = parser.parse_known_args()
    return args

# %%
if __name__ == '__main__':
    try:
        params = vars(get_params())
        if params['city']=="tangshan":
            params['n_global_feats'] = 15
            params['n_neighbors'] = 3
        else:
            params['n_global_feats'] = 7
            params['n_neighbors'] = 4
        logging.info((params['target'],params['city']))
        print(params)
        train(params)
    except Exception as exception:
        logging.exception(exception)
        raise