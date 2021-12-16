# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [ ]
#---------------------------------------------------------------------------
# basic-module
import matplotlib.pyplot as plt
import sys,os,re,glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.simplefilter('ignore')
from tqdm import tqdm
import seaborn as sns
#---------------------------------------------------
# sori -module
sys.path.append('/home/ysorimachi/tool')
from utils_log import log_write #(path,mes,init=False)
#---------------------------------------------------
import subprocess
from tool_time import dtinc
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
mms = MinMaxScaler()

sys.path.append("/home/ysorimachi/work/synfos/py/som_data")
from a01_99_utils import *
from c01_som_cluster import *
from x99_pre_dataset import load_rad, load_10

from util_Model2 import Resid2
import pickle

from layers import LeNet

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    from torch.utils.data import DataLoader, TensorDataset, Dataset
    from torchvision import transforms
    from torchinfo import summary #torchinfoはニューラルネットの中身を見れるのでおすすめ
except:
    print("Not Found Troch Modules ...")


# DHOME="/work2/ysorimachi/mix_som/dat" #synfos/ data 

# OHOME="/work2/ysorimachi/mix_som/out/syn_data/data"
# OHOME="/work2/ysorimachi/mix_som/out/syn_mesh/data"
DHOME="/work2/ysorimachi/mix_som/out/syn_mesh/data"
DHOME2="/work2/ysorimachi/mix_som/out/syn_mesh/data2" #保存先

SYNFOS_INIT = "../../src/now_jst_som.txt"
CLUSTER="/home/ysorimachi/data/synfos/som/model/cluster"
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 18
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


ERR="/home/ysorimachi/data/synfos/som/model/err"
CATE="/home/ysorimachi/data/synfos/cate"

#------------2021.10.08 --------------
def save_numpy(save_path,obj):
    save_path2 = save_path.split(".")[0]
    np.save(save_path2, obj.astype('float32'))
    return 

def load_numpy(path):
    obj = np.load(path)
    return obj

def load_numpy2(path):
    obj = np.load(path)
    N,H,W = obj.shape
    obj = obj.reshape(H,W,N) #image　読み込みの為に順番変更 2021.10.26
    return obj

def save_model(path,model):
  with open(path,"wb") as pkl:
    pickle.dump(model,pkl)
  return
def load_model(path):
  with open(path,"rb") as pkl:
    model = pickle.load(pkl)
  return model

def load_cate(n=1, train="ALL"):
    path = glob.glob(f"/home/ysorimachi/data/synfos/cate/DAY_cate{n}*.csv")[0]
    df = pd.read_csv(path)
    df = df.sort_values("dd").reset_index(drop=True)
    
    df["time"] = df["dd"].apply(lambda x: pd.to_datetime(f"{x}0000"))
    df = train_flg(df)
    df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
    df = df.drop(["time"],axis=1)
    
    if n==1:
        cname = "DAY_MAX_CATE"
    elif n==2:
        cname = "DAY_CI"
    
    if train == "ALL":
        _dd = df["dd"].values
        _ll = df[cname].values
        return _dd,_ll
    else:
        if train:
            df = df[df["istrain"]==1]
        else:
            df = df[df["istrain"]==0]
        _dd = df["dd"].values
        _ll = df[cname].values
        return _dd,_ll



def DataLoader2(_CATE,name,n_cate=1,batch_size=4):
    """
    参考情報：https://pystyle.info/pytorch-how-to-create-custom-dataset-class/
    複数データセットを持つ場合の作成方法 -> http://kaga100man.com/2019/03/25/post-102/
     
    """
    
    def img_tensor(train=True):
        _dd,_ll = load_cate(n_cate,train=train)
        _path,_dd2,_ll2 =[],[],[]
        DHOME="/work2/ysorimachi/mix_som/out/syn_mesh/data2"
        
        for dd,ll in zip(_dd,_ll):
            path = os.path.join(DHOME,f"{dd}_{name}.npy")
            if os.path.exists(path):
                _path.append(path),_dd2.append(dd),_ll2.append(ll)
        
        _img=[]
        for path in _path:
            img = load_numpy(path)
            _img.append(img)
        img = np.stack(_img,axis=0)
        
        #numpy -> tensor
        
        lbl = torch.LongTensor(_ll2) #大きな値の分類について(整数型で格納して利用するというもの)
        img = torch.Tensor(img)
        return img,lbl
    
    # main 
    img_train,lbl_train = img_tensor(train=True)
    img_test,lbl_test  = img_tensor(train=False)
    
    train_set = TensorDataset(img_train,lbl_train)
    test_set = TensorDataset(img_test,lbl_test)
    # print("train-size = ", len(_dd0)," -> ",len(train_set))
    # print("test-size = ", len(_dd1)," ->  ",len(test_set))
    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    
    return train_loader,test_loader


FITTING="/home/ysorimachi/data/synfos/cate/fitting"
PREDICT="/home/ysorimachi/data/synfos/cate/predicting"
def fitting(_CATE,name,n_cate,N_EPOCH=10):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    save_path = f"{FITTING}/Lenet_{n_cate}_{name}.pkl"
    csv_path = f"{FITTING}/Study_{n_cate}_{name}.csv"
    
    # data loader making ---
    train_loader,test_loader = DataLoader2(_CATE,name,n_cate,batch_size=4)
    print(datetime.now(), "[END]", "prepaired DATASET !")
    # net  making ---
    if n_cate ==1:
        out_size = 8
    elif n_cate ==2:
        out_size = 5
        
    n_channel = len(_CATE)
    net = LeNet(n_channel=n_channel, out_size=out_size)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9)
    # Fit setting ---
    _l0,_l1,_ac1 = [],[],[]
    
    def train(data_loader,device="cpu"):
        net.train()
        run_loss = 0
        for _img,_lbl in data_loader:
            _img.to(device)
            _lbl.to(device)
            optimizer.zero_grad()
            _out = net(_img)
            
            # print(_out.shape)
            # print(_lbl.shape)
            # sys.exit()
            # sys.exit()
            loss = criterion(_out,_lbl)
            run_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = run_loss / len(data_loader)
        return train_loss
    
    def valid(data_loader,device="cpu"):
        net.eval()
        run_loss=0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for _img,_lbl in data_loader:
                _img.to(device)
                _lbl.to(device)
                optimizer.zero_grad()
                _out = net(_img)
                loss = criterion(_out,_lbl)
                run_loss += loss.item()
                #pred
                _pred = _out.max(1, keepdim=True)[1]
                _lbl = _lbl.view_as(_pred)
                correct += _pred.eq(_lbl).sum().item()
                total += _lbl.size(0)
        
        val_loss = run_loss / len(data_loader)
        val_acc = correct / total
        return val_loss, val_acc
    
    print(datetime.now(), "[START]", f"train ({N_EPOCH})!")
    print("-"*50)
    for epoch in range(N_EPOCH):
        l0 = train(train_loader, device=device)
        l1,ac1 = valid(test_loader, device=device)
        
        print(f"EPOCH:{epoch+1} -> TrainLoss={l0:.3f} | TestLoss={l1:.3f} | ACC={ac1*100:.3f}")
        _l0.append(l0), _l1.append(l1),_ac1.append(ac1)
    print(datetime.now(), "[END]", f"train ({N_EPOCH})!")
    
    save_model(save_path,net) #model pickle save ---
    df = pd.DataFrame()
    df["train_loss"] = _l0
    df["test_loss"] = _l1
    df["test_acc"] = _ac1
    df.to_csv(csv_path, index=False)
    return 
            
def predicting(_CATE,name,n_cate):

    _dd,_ll = load_cate(n_cate,train="ALL")
    
    net = load_model(f"/home/ysorimachi/data/synfos/cate/fitting/Lenet_{n_cate}_{name}.pkl")
    DHOME="/work2/ysorimachi/mix_som/out/syn_mesh/data2"
    csv_path =f"{PREDICT}/pred_{n_cate}_{name}.csv"
    
    _d2,_l2,_p2 = [],[],[]
    
    weight_hash = {}
    with torch.no_grad():
        
        for dd,ll in zip(_dd,_ll):
            path = os.path.join(DHOME,f"{dd}_{name}.npy")
            if os.path.exists(path):
                img = load_numpy(path)
                C,H,W = img.shape
                
                img = torch.Tensor(img)
                img = img.view(1,C,H,W)
                
                out= net(img)
                # _,predicted = torch.max(out,1)
                pred  = out.detach().numpy().tolist()
                pred_weight = F.softmax(out)
                weight = pred_weight.detach().numpy().tolist()
                
                weight_hash[dd] = weight
                _l2.append(ll)
    
    df = pd.DataFrame(weight_hash).T
    df.index.name = "dd"
    df["ll"] = _l2
    df.columns = ["pred","true"]
    df.to_csv(csv_path)
    return

if __name__== "__main__":
    # make_cate_rule() #cate rulr の作成
    
    #cate ---
    _FD,_CATE,name=["FD2","FD2","FD2"],["HICA","MICA","LOCA"],"cloud3"
    # _FD,_CATE,name=["FD1","FD2"],["MSPP","MSPP"],"sp_Ts2"
    
    for n_cate,N_EPOCH in zip([1,2],[4,3]):
        fitting(_CATE,name,n_cate=n_cate,N_EPOCH=N_EPOCH)
        predicting(_CATE,name,n_cate=n_cate)
    
    #---cehck---
    # check()
    
        