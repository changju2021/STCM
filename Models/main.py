# %%
import numpy as np
import torch
from load_data import *
from utils import *
import os
from tqdm import tqdm
import logging
import pickle
import argparse
from STCM import *
torch.manual_seed(2333)
torch.cuda.manual_seed(2333)
np.random.seed(2333)
torch.backends.cudnn.deterministic = True  
# %%
def train(args):
    save_path = 'models/noise'
    model_path = save_path+'/'+args['city']+args['target']+'.pt'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda"if torch.cuda.is_available()else"cpu")
    model = STCM(args).to(device)
    
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
    model = STCM(args).to(device)
    model.load_state_dict(torch.load(model_path))
    # val_loss = evaluate_model(model, loss, val_iter)
    evaluate_metric(model,test_iter,scaler)


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
# %%
