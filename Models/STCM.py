# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
# %%
class STCM(nn.Module):
    def __init__(self,hps):
        super(STCM,self).__init__()
        self.hps = hps
        self.droput = nn.Dropout(hps['dropout'])
        self.enc1 = nn.GRU(1,self.hps['hid1'],batch_first=True)
        
        self.enc2 = nn.GRUCell(self.hps['n_neighbors'],self.hps['hid2'])
        self.wg = nn.Linear(self.hps['hid1']+self.hps['hid2'],self.hps['ga'])
        self.ug = nn.Linear(self.hps['n_global_feats']+1,1,False)
        self.vg = nn.Linear(self.hps['l1'],self.hps['ga'],False)
        self.W = nn.Linear(self.hps['ga'],1,False)
    
        self.wgg = nn.Linear(self.hps['hid1']+self.hps['hid2'],self.hps['ta'])
        self.V = nn.Linear(self.hps['ta'],1)
        
        self.fuse = nn.Linear(self.hps['hid1']+self.hps['hid2'],self.hps['fuse'])
        self.dec = nn.GRUCell(self.hps['n_neighbors'],self.hps['fuse'])
        self.fc = nn.Linear(self.hps['fuse'],1)

    def globalAttn(self,s,h,x,x_int,getAttn=False):
        # s(n,hid2) h(n,hid1),x (n,24,num_mobile,in_features) ,x_int (n,num_mobile)
        # print(s.shape,h.shape,x.shape,x_int.shape)
        sh = self.wg(torch.cat((s,h),dim=1)) #(n,hid2)
        # print(self.ug(x).shape)
        t = self.vg(self.ug(x).squeeze(3).permute(0,2,1))
        # print(t.shape)
        sh = sh.unsqueeze(1).repeat(1,self.hps['n_neighbors'],1)
        # print(sh.shape)
        a = self.W(F.tanh(t+sh)).squeeze(2)
        if getAttn:
            return x_int*a,F.softmax(a,dim=1)
        return x_int*a
    
    def temporalAttn(self,ht,s):
        # ht (n,168,hid1) s (n,hid2)
        timestamps = list(range(self.hps['l2']-7,self.hps['l2']-1))
        timestamps += list(range(self.hps['l2']-24,-1,-24))
        ht = ht[:,timestamps,:] #n,13,hid1
        s = s.unsqueeze(1).repeat(1,len(timestamps),1) #n,13,hid2
        t = self.wgg(torch.cat((ht,s),dim=2))# n,13,ga
        e = self.V(F.tanh(t)) # n,13,1
        return torch.sum(ht*e,dim=1)
    
    def forward(self,x,getGA=False,getTA=False):
        # mdatax [128, 24, num_mobile, 15]
        # mdatay [128, 24, num_mobile]
        # sdatay [128, 168]
        batch_size = x[0].size(0)
        # long-term encoder
        ht,_ = self.enc1(x[2].unsqueeze(dim=2)) # ht [n,168,hid1]
        
        s = torch.zeros((batch_size,self.hps['hid2']),dtype=x[0].dtype,device=x[0].device)
        x_m = torch.cat((x[0],x[1].unsqueeze(dim=3)),dim=3) # x_m [n,24,num_mobile,n_global_feats+1]
        
        x_tilde, gAttn = None, None
        
        for t in range(-self.hps['l1'],0,1):
            if t == -1 and getGA:
                x_tilde, gAttn = self.globalAttn(s,ht[:,t,:],x_m,x[1][:,t,:],getAttn=True)
                
            x_tilde = self.globalAttn(s,ht[:,t,:],x_m,x[1][:,t,:]) # (128,num_mobile)
            s = self.enc2(x_tilde,s)
        
        d = self.temporalAttn(ht,s) # n,hid1
        hf = self.fuse(torch.cat((d,s),dim=1))
        out = self.dec(x_tilde,hf)
        out = F.sigmoid(self.fc(out))
        if getGA:
            return gAttn
        else:
            return out #(n,1)
        
#%%
# from params import hps
# device = torch.device("cuda"if torch.cuda.is_available()else"cpu")
# x=[torch.randn(128, 24, 3,15).to(device),torch.randn(128, 24, 3).to(device),torch.randn(128, 168).to(device)]
# model = MutiAttnV4(hps).to(device)
# out=model(x)
# print(out.shape)
# %%

# %%
