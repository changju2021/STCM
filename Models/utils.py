import torch
import numpy as np
import matplotlib.pyplot as plt



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

def evaluate_metric(model, data_iter, scaler, isreturn=False):
    model.eval()
    y_true, y_pred= [],[]
    with torch.no_grad():
        for x, y in data_iter:
            y_true += scaler.inverse_transform(y.cpu().numpy().reshape(-1,1)).reshape(-1).tolist()
            y_pred += scaler.inverse_transform(model(x).cpu().numpy().reshape(-1,1)).reshape(-1).tolist()
        y_true, y_pred = np.asarray(y_true),np.asarray((y_pred))
        d = np.abs(y_true - y_pred)
        MAE = round(d.mean(),3)
        RMSE = round(np.sqrt((d**2).mean()),3)
        SMAPE = round((2*d / (np.abs(y_true)+np.abs(y_pred))).mean()*100,3)
        
        mask = y_true!=0
        d = np.abs(y_true[mask]-y_pred[mask])
        MAPE = round(( d / np.abs(y_true [mask])).mean()*100,3)
        print(MAE,MAPE,RMSE,SMAPE)
        if isreturn:
            return y_true,y_pred
def getGlobalAttn(model, data_iter):
    model.eval()
    gAttn = None
    with torch.no_grad():
        for x,y in data_iter:
            tmp = model(x,getGA=True).cpu().numpy()
            if gAttn is None:
                gAttn = tmp
            else:
                gAttn = np.concatenate((gAttn,tmp),axis=0)
    return gAttn
# def evaluate_metric(model, data_iter, scaler): # 返回注意力
#     y_true,prediction = [],[]
#     model.eval()
#     with torch.no_grad():
#         mae, mape, mse,smape = [], [], [], []
#         gAttn,tAttn = [],[]
#         for x, y in data_iter:
#             y = scaler.inverse_transform(y.cpu().numpy().reshape(-1,1)).reshape(-1)
#             y_pred,ga,ta = model(x,False)
#             gAttn+=ga.data.cpu().numpy().tolist()
#             tAttn+=ta.data.cpu().numpy().tolist()
#             y_pred = scaler.inverse_transform(y_pred.cpu().numpy().reshape(-1,1)).reshape(-1)
#             d = np.abs(y - y_pred)
#             mae += d.tolist()
#             mape += (d / np.abs(y)).tolist()
#             mse += (d ** 2).tolist()
#             smape += (2*d / (np.abs(y)+np.abs(y_pred))).tolist()
#             y_true += y.tolist()
#             prediction += y_pred.tolist()       
#         MAE = round(np.array(mae).mean(),3)
#         MAPE = round(np.array(mape).mean()*100,3)
#         RMSE = round(np.sqrt(np.array(mse).mean()),3)
#         SMAPE = round(np.array(smape).mean()*100,3)
#         gAttn = np.array(gAttn)#n,4
        # tAttn = np.array(tAttn)#n,12
        # prediction = np.array(prediction)
        # y_true = np.array(y_true)
        # print(MAE,MAPE,RMSE,SMAPE)
        #print(gAttn.shape,tAttn.shape)
       # compare(y_true,prediction,)
        #write(str(model.parameters),{'MAE':MAE,'MAPE':MAPE,'RMSE':RMSE,'SMAPE':SMAPE})
        # return prediction,y_true,gAttn,tAttn
        
        
# def compare(y_pred,y_true,gAttn,tAttn):
#     colors = []
#     labels = ['749','769','801','842']
#     #fig = plt.subplot(2,1,1)
#     plt.plot(y_true,color='g')
#     plt.plot(y_pred,color='r')
#     plt.bar(gAttn[:,0], align="center", color=colors[0],label=labels[0])
#     for i in range(1,4):
#         plt.bar(gAttn[:,i], align="center", bottom=np.sum(gAttn[:,:i],dim=1), color=colors[i], label=labels[i])
#     plt.xlabel('Time')
#     plt.ylabel(target)

    # fig = plt.subplot(2,1,2)
    # plt.plot(y_true,color='g')
    # plt.plot(y_pred,color='r')
    # plt.bar(gAttn[:,0], align="center", color=colors[0],label=labels[0])
    # for i in range(1,12):
    #     plt.bar(tAttn[:,i], align="center", bottom=np.sum(tAttn[:,:i],dim=1), color=colors[i])
    # plt.xlabel('Time')
    # plt.ylabel(target)
    
    
    
# def write(parameters,mdict):
#     # model.parameters
#     with open(result_path,'a+') as f:
#         f.write("==============================================\n")
#         f.write(target)
#         sents = ''
#         for k,v in mdict.items():
#             sents += k+':'+str(v)+'   '
#         f.write(sents+'\n')
#         f.write(parameters)
#         f.write('\n')  
    