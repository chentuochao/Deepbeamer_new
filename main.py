import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from loss import FuseLoss
from Generative_model import Earbud_Net
from TasNet import ConvTasNet
from Baseline import Base_Model
from prepare_data2 import generate_data

import time
from util import calculate_gain
import numpy as np
# ------------------- model init ------------------
BATCH=40
batch_mul=30

HIGH_PROB=1.0
LOW_PROB=0.3

cuda_id=0

model = "baseline"

if model == "Unet":
    generative_model = Earbud_Net(num_class = 2)
elif model == "TasNet":
    '''
    N: Number of filters in autoencoderi
    L: Length of the filters (in samples)
    B: Number of channels in bottleneck 1 × 1-conv block
    H: Number of channels in convolutional blocks
    P: Kernel size in convolutional blocks
    X: Number of convolutional blocks in each repeat
    R: Number of repeats
    C: Number of speakers
    norm_type: BN, gLN, cLN
    causal: causal or non-causal
    '''
    B, H, P, X, R, C, norm_type, causal = 32, 64, 3, 5, 2, 1, "gLN", False
    N, L = 32, 128
    Class_num = 2
    generative_model = ConvTasNet(N, L, B, H, P, X, R, C, Class_num, norm_type=norm_type)
elif model == "baseline":
    M, N, L, T, B = 15, 32, 256, 128000+128, 64
    Class_num = 2
    generative_model = Base_Model(L, N, B, Class_num)
else:
    print("Invalid model name")
    raise KeyboardInterrupt


loss_net=FuseLoss(2400, 10)

#generative_model=torch.nn.DataParallel(generative_model, device_ids=[0,1])
#loss_net=torch.nn.DataParallel(loss_net, device_ids=[0,1])

generative_model=generative_model.cuda(cuda_id)
loss_net=loss_net.cuda(cuda_id)

#-------------prepare data ----------------------------------------------

train_dataset, test_dataset = generate_data(if_TEST = 0, seg_length = 3, TRAIN_NUM = 6000, TEST_NUM= 800)#TRAIN_NUM = 5400, TEST_NUM= 600)
#print(len(train_dataset))
#print(len(test_dataset))
index = 0
#for d in train_dataset:
#    print(index)
#    index += 1
#    print(d)
#    if index > 5600: break
#raise KeyboardInterrupt
dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=4)
testloader=torch.utils.data.DataLoader(test_dataset, batch_size=BATCH, shuffle=False, num_workers=4)

# ------------------- model training and testing function----------------

def train_epoch(model, lossmodel, optimizer, scheduler, dataloader, save_path=None, last_loss=None):
    model.train()
    losses=[]
    tick0=time.time()
    
    optimizer.zero_grad()
    
    batch_losses=[]
    nan=False
    for batch_idx, (total, select_vector, gt) in enumerate(dataloader):
        #print(batch_idx*5)i
        total = total[:, :, :T]
        gt = gt[:, :, :T]
        tick1=time.time()
        #print(total.size(), select_vector.size(),gt.size())
        total = total.float()
        select_vector = select_vector.float()
        gt = gt.float()
        data=total.cuda(cuda_id)
        gt=gt.cuda(cuda_id)
        select_vector=select_vector.cuda(cuda_id)

        tick2=time.time()
        output=model(data, select_vector)
        #print(output) 
        loss=lossmodel(output, gt).sum()
        #print(loss, loss.sum())
        l=loss.item()
        #print(l)
        r=np.random.random()
        loss.backward()
        
        tick3=time.time()
        batch_losses.append(l)
        if batch_idx%batch_mul==batch_mul-1:
            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            
            # check nan
            if not np.isnan(batch_losses).any():
                optimizer.step()
                print("Batch %d/%d, Avg Train Loss: %f" %(batch_idx*BATCH, 5000, np.mean(batch_losses)))
            else:
                print("NaN loss, skip")
                nan=True
                break
            
            optimizer.zero_grad()
            batch_losses=[]
            
        
        data=None
        premix=None
        losses.append(l)
        
        tick0=tick3
        
    scheduler.step()
    print("scheduler lr: ", scheduler.get_last_lr()[0])
    #print("selective back: ", low_cnt, high_cnt)
    
    if not nan:
        return np.mean(losses)
    else:
        return None

# evaluate
def test_epoch(model, lossmodel, testloader):
    model.eval()
    gains=[]
    losses=[]
    inputgains=[]
    with torch.no_grad():
        for batch_idx, (total, select_vector, gt) in enumerate(testloader):
            total = total[:, :, :T]
            gt = gt[:, :, :T]
 
            total = total.float()
            select_vector = select_vector.float()
            gt = gt.float()
            data=total.cuda(cuda_id)
            gt_gpu=gt.cuda(cuda_id)
            select_vector=select_vector.cuda(cuda_id)
                
            output=model(data, select_vector)
            
            loss=lossmodel(output, gt_gpu).sum()
            losses.append(loss.item())
            # test si-sdr
            output_cpu=output.cpu().detach().numpy()
            for i in range(output_cpu.shape[0]):
                if torch.max(gt[i,0])>0.02:
                    gain, _,_=calculate_gain(gt[i, 0, 24000:], total[i, 0,24000:], output_cpu[i, 0,24000:])
                    gains.append(gain)
            #        gain, _,_=calculate_gain(premix[i,0], total[i,0], bfdata[i,1])
            #        inputgains.append(gain)
            
    return losses, gains#, inputgains

def load_path(model, path):
    model.load_state_dict(torch.load(path), strict=False)

# ----------------- begin training -------------

if_train = True
model_path = '../trained_model/neweval-tasnet.bin'

if if_train:
    from torch.optim.lr_scheduler import CosineAnnealingLR
    opt1=optim.AdamW(generative_model.parameters(), lr=3e-4) # 2e-4
    opt2=optim.AdamW(generative_model.parameters(), lr=1e-3) # 2e-3
    opt3=optim.AdamW(generative_model.parameters(), lr=3e-4) # 5e-4

    opts=[(opt1, StepLR(opt1, step_size=40, gamma=0.7), 2),
          (opt2, CosineAnnealingLR(opt2, T_max=8, eta_min=1e-4), 24),
          (opt3, StepLR(opt3, step_size=100, gamma=0.6), 100)]
    train_losses=[]
    test_losses=[]
    best_gain = -100

    for (opt, sch, n) in opts:
        for i in range(n):
            print("epoch"+str(i)+": " +str(len(train_losses)))
            last_loss=None
            res=train_epoch(generative_model, loss_net, opt, sch, dataloader, model_path, last_loss)
            losses, gains=test_epoch(generative_model, loss_net, testloader)
            test_losses.append(np.mean(losses))
            test_gain = np.mean(gains)
            print("train avg loss: %f\n, test avg loss: %f, test avg gain %f"%(res, np.mean(losses), test_gain))
            if res is not None:
                train_losses.append(res)
            else:
                print("Warning, the training loss is None")
                break
            
            if model_path is not None:
                if test_gain > best_gain:
                    best_gain = test_gain
                    torch.save(generative_model.state_dict(), model_path)
                    print("model " + model_path + " saved")
else:
    load_path(generative_model, model_path)
    losses, gains=test_epoch(generative_model, loss_net, testloader)
    print(gain)
    print("test avg loss: %f, test avg gain %f", res, np.mean(losses), np.mean(gains))
    
