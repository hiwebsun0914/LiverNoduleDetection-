# Code of the paper Cross-Attention Guided Loss-Based Deep Dual-Branch Fusion Network for Liver Tumor Classification
# If any question, please contact the author.
# https://github.com/Wangrui-berry/Cross-attention.

import torch
from cmath import exp
import os
from site import abs_paths
import sys
import numpy as np

from feeder_8modal_7class import MRIDataset

from model.crossatten import generate_crossatten
from model.ablation.Rnet import generate_rnet
from model.ablation.PA_Net import generate_panet
# IA_Net -> crossatten
from model.fuxian.MIL.deepganet import generate_deepganet
from model.fuxian.MIL.DAMIDL8m_semi import DAMIDL
from model.fuxian.MIL.Resmil import generate_resmil


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import time
import datetime
starttime = datetime.datetime.now()+datetime.timedelta(hours=8)
starttime = starttime.strftime("%Y-%m-%d_%H_%M_%S")

import argparse
import yaml
syspath = os.path.dirname(os.path.abspath(__file__))


import json
from monai.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter 


from utils.losses import BCEFocalLoss, MultiCEFocalLoss
from utils.losses import CrossEntropyLoss as WeightCE
device_now = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

#######################loss-based attention(P/I weight)################
def rampup(global_step, rampup_length=24):
    if global_step <rampup_length:
        global_step = float(global_step)
        rampup_length = float(rampup_length)
        phase = 1.0 - np.maximum(0.0, global_step) / rampup_length
    else:
        phase = 0.0
    return np.exp(-5.0 * phase * phase)


def format_duration(seconds):
    return str(datetime.timedelta(seconds=int(max(0, seconds))))

def train(train_loader, model, lossCE, loss_patch, optimizer, scaler, args, epoch_idx):
    start = time.time()
    model.train()
    train_loss = 0.0
    correct = 0.0
    seen = 0
    train_pre = 0.0
    train_rec = 0.0
    train_f1 = 0.0

    rampup_value = rampup(epoch_idx)
    u_w = 0 if epoch_idx == 0 else 0.2 * rampup_value
    u_w = torch.tensor([u_w], dtype=torch.float32, device=device_now, requires_grad=False)

    train_pbar = tqdm(
        enumerate(train_loader, start=1),
        total=len(train_loader),
        desc=f"Train {epoch_idx + 1}/{args.num_epoch}",
        leave=False,
    )

    for step, (images_t2, images_dwi, images_in, images_out, images_pre, images_ap, images_pvp, images_dp, label) in train_pbar:
        images_t2, images_dwi, images_in, images_out, images_pre, images_ap, images_pvp, images_dp, labels = images_t2.to(device_now,non_blocking=True), images_dwi.to(device_now,non_blocking=True), images_in.to(device_now,non_blocking=True), images_out.to(device_now,non_blocking=True), images_pre.to(device_now,non_blocking=True), images_ap.to(device_now,non_blocking=True), images_pvp.to(device_now,non_blocking=True), images_dp.to(device_now,non_blocking=True), label.to(device_now,non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(args.amp and device_now.type == "cuda")):
            bag_preds, out1, f_compose, alpha_compose, out2, f_patch_com, alpha_patch_com = model(images_t2.to(torch.float32),images_dwi.to(torch.float32), images_in.to(torch.float32), images_out.to(torch.float32), images_pre.to(torch.float32), images_ap.to(torch.float32), images_pvp.to(torch.float32), images_dp.to(torch.float32))
            loss_1 = lossCE(bag_preds, labels.long())
            patch_targets = labels.repeat(4*4,1).permute(1,0).contiguous().view(-1)
            loss_2_R = loss_patch(f_compose, patch_targets, weights=alpha_compose.view(-1))
            loss_2_R_alphapatchcom = loss_patch(f_compose, patch_targets, weights=alpha_patch_com.view(-1))
            loss_2_P_alphacom = loss_patch(f_patch_com, patch_targets, weights=alpha_compose.view(-1))
            loss_2_P = loss_patch(f_patch_com, patch_targets, weights=alpha_patch_com.view(-1))
            simlabel = torch.ones(alpha_compose.shape[0], device=device_now)
            sim_loss = torch.nn.CosineEmbeddingLoss()(alpha_compose,alpha_patch_com,simlabel)
            loss = loss_1 + u_w*loss_2_P/bag_preds.size(0) + u_w*loss_2_R/bag_preds.size(0) + 0.25*u_w*loss_2_R_alphapatchcom + 0.25*u_w*loss_2_P_alphacom + 0.25*sim_loss

        _, predicted = torch.max(bag_preds.data, 1)
        correct += predicted.eq(labels).sum().float().item()
        seen += labels.size(0)
        labels_cpu = labels.detach().cpu()
        predicted_cpu = predicted.detach().cpu()
        train_pre += metrics.precision_score(labels_cpu.tolist(), predicted_cpu.tolist(), average='micro')
        train_rec += metrics.recall_score(labels_cpu.tolist(), predicted_cpu.tolist(), average='micro')
        train_f1 += metrics.f1_score(labels_cpu.tolist(), predicted_cpu.tolist(), average='micro')

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_pbar.set_postfix(loss=f"{(train_loss / step):.4f}", acc=f"{(correct / seen):.4f}")

    train_loss = train_loss / len(train_loader)
    train_acc = correct / len(train_loader.dataset)
    train_f1 = train_f1 / len(train_loader)
    return train_loss, train_acc, train_f1, time.time() - start


def validate(val_loader, model, lossCE, args, epoch_idx):
    start = time.time()
    model.eval()

    val_loss = 0.0
    correct = 0.0
    seen = 0
    test_pre = 0.0
    test_rec = 0.0
    val_f1 = 0.0

    val_pbar = tqdm(
        enumerate(val_loader, start=1),
        total=len(val_loader),
        desc=f"Val   {epoch_idx + 1}/{args.num_epoch}",
        leave=False,
    )

    with torch.no_grad():
        for step, (images_t2, images_dwi, images_in, images_out, images_pre, images_ap, images_pvp, images_dp, labels) in val_pbar:
            images_t2, images_dwi, images_in, images_out, images_pre, images_ap, images_pvp, images_dp, labels = images_t2.to(device_now,non_blocking=True), images_dwi.to(device_now,non_blocking=True), images_in.to(device_now,non_blocking=True), images_out.to(device_now,non_blocking=True), images_pre.to(device_now,non_blocking=True), images_ap.to(device_now,non_blocking=True), images_pvp.to(device_now,non_blocking=True), images_dp.to(device_now,non_blocking=True), labels.to(device_now,non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(args.amp and device_now.type == "cuda")):
                outputs,_,_,_,_,_,_ = model(images_t2.to(torch.float32),images_dwi.to(torch.float32), images_in.to(torch.float32), images_out.to(torch.float32), images_pre.to(torch.float32), images_ap.to(torch.float32), images_pvp.to(torch.float32), images_dp.to(torch.float32))
                loss = lossCE(outputs, labels.long())

            val_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().float().item()
            seen += labels.size(0)

            labels_cpu = labels.detach().cpu()
            preds_cpu = preds.detach().cpu()
            test_pre += metrics.precision_score(labels_cpu.tolist(), preds_cpu.tolist(), average='micro')
            test_rec += metrics.recall_score(labels_cpu.tolist(), preds_cpu.tolist(), average='micro')
            val_f1 += metrics.f1_score(labels_cpu.tolist(), preds_cpu.tolist(), average='micro')
            val_pbar.set_postfix(loss=f"{(val_loss / step):.4f}", acc=f"{(correct / seen):.4f}")

    val_loss = val_loss / len(val_loader)
    val_acc = correct / len(val_loader.dataset)
    val_f1 = val_f1 / len(val_loader)
    return val_loss, val_acc, val_f1, time.time() - start


if __name__ == '__main__':

    print("Start experiment:", starttime)
    seed = 5
    print("seed_everything:",seed)
    from monai.utils import set_determinism
    set_determinism(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True,warn_only=True)
    
    parser = argparse.ArgumentParser()
# Dataset parameters
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--csv_path', default='', type=str, 
                        help='contain both train and val data')
    parser.add_argument('--num_classes', type=int, default=7, metavar='N',
                    help='number of label classes (Model default if None)')
    parser.add_argument('--result_dir', default='', type=str, 
                        help='contain both train and val data')
    parser.add_argument('--json_dir', default='', type=str, 
                        help='contain both train and val data')
        
# Model parameters
    parser.add_argument('--net', type=str, default="crossatten", 
                        help='brain,DAMIDL,deepganet,Rbase, RANet,PANet,crossatten]')
    parser.add_argument('--cuda',type=str_to_bool,default=True, 
                        help='load in GPU')
#Pretrained model
    parser.add_argument('--resume',type=str_to_bool,default=False,
                        help='read model.state.dict()')
    parser.add_argument('--weight_path',default='',help='2023-04-03_23_51_03_epoch500; load model path: results/2023-03-29_15_59_04.pth')  
#Parallel
    #Parser.add_argument('--device',type=str,default='cuda:0',help='')
    parser.add_argument('--data_parallel', type=str_to_bool, default=False)
    # parser.add_argument('--device_id',type=list,default=[0,1],help='')
#Parameter
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='batch size for dataloader')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                        help='weight decay rate')
    parser.add_argument('--num_epoch', type=int, default=65, metavar='N',
                    help='number of epochs to train (default: 300)')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
    
    parser.add_argument('--n_fold', type=str, default='5fold', 
                        help='1fold/5fold')
    parser.add_argument('--amp', type=str_to_bool, default=True,
                        help='enable mixed precision training to reduce GPU memory')
    args = parser.parse_args()
    device_now = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    print("device_now:", device_now)
    print(
        f"config: net={args.net} fold={args.n_fold} bs={args.batch_size} "
        f"epochs={args.num_epoch} lr={args.lr} resume={args.resume} amp={args.amp}"
    )

    writer = SummaryWriter(log_dir=syspath+"/log/"+starttime[:10]+"/"+starttime[11:],flush_secs=60)
    os.makedirs(args.result_dir, exist_ok=True)

    last_ckpt_path = os.path.join(args.result_dir, "last.pth")
    best_ckpt_path = os.path.join(args.result_dir, "best.pth")

    if args.net == "Rbase":
        model = generate_rnet(10)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)



    elif args.net =="deepganet":
        model = generate_deepganet(18)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    elif args.net =="DAMIDL":
        model = DAMIDL(patch_num=4, feature_depth=None, num_classes=7)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    elif args.net == "3dres":
        model = generate_crossatten(10)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    model = model.to(device_now,non_blocking=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.333, last_epoch= -1)

    #######################loss-CE###############
    # lossCE= torch.nn.CrossEntropyLoss()   # if binary classification
    lossCE= MultiCEFocalLoss(class_num=7,device_now=device_now)
    #######################loss-based attention###############
    loss_patch = WeightCE(aggregate='sum',device_now=device_now)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device_now.type == "cuda"))

    train_dataset = MRIDataset(args, flag='Train')
    val_dataset = MRIDataset(args, flag='Val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=6, worker_init_fn=np.random.seed(seed),pin_memory=True)   #worker_init_fn=np.random.seed(seed)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=6,worker_init_fn=np.random.seed(seed),pin_memory=True)       #worker_init_fn = worker_init_fn(42),
  
    start_epoch = 0
    best_epoch = 0
    best_acc = 0.0
    best_f1 = 0.0
    best_acc_f1 = 0.0

    if args.resume:
        resume_path = args.weight_path if args.weight_path else last_ckpt_path
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device_now)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if 'scaler' in checkpoint and checkpoint['scaler'] is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device_now)
        start_epoch = checkpoint.get('epoch', 0)
        best_epoch = checkpoint.get('best_epoch', 0)
        best_acc = checkpoint.get('best_acc', 0.0)
        best_f1 = checkpoint.get('best_f1', 0.0)
        best_acc_f1 = checkpoint.get('best_acc_f1', 0.0)
        print(f"Resume from epoch {start_epoch}/{args.num_epoch}, best_acc={best_acc:.4f}")

    train_wall_start = time.time()
    epochs_pbar = tqdm(range(start_epoch, args.num_epoch), desc="Epochs", unit="epoch")
    for epoch in epochs_pbar:
        scheduler.step()
        train_loss, train_acc, train_f1, train_sec = train(train_loader, model, lossCE, loss_patch, optimizer, scaler, args, epoch)
        val_loss, val_acc, val_f1, val_sec = validate(val_loader, model, lossCE, args, epoch)

        writer.add_scalar('train/train_loss', train_loss, epoch)
        writer.add_scalar('train/train_acc', train_acc, epoch)
        writer.add_scalar('val/val_loss', val_loss, epoch)
        writer.add_scalar('val/val_acc', val_acc, epoch)
        writer.add_scalar('train/train_f1', train_f1, epoch)
        writer.add_scalar('val/val_f1', val_f1, epoch)

        if val_f1 > best_f1:
            best_f1 = val_f1
        is_new_best = val_acc > best_acc
        if is_new_best:
            best_acc = val_acc
            best_acc_f1 = val_f1
            best_epoch = epoch + 1

        ckpt = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict() if scaler.is_enabled() else None,
            'best_epoch': best_epoch,
            'best_acc': best_acc,
            'best_f1': best_f1,
            'best_acc_f1': best_acc_f1,
            'args': vars(args),
            'starttime': starttime,
        }
        torch.save(ckpt, last_ckpt_path)
        if is_new_best:
            torch.save(ckpt, best_ckpt_path)

        epochs_done = epoch - start_epoch + 1
        elapsed = time.time() - train_wall_start
        avg_epoch = elapsed / max(1, epochs_done)
        remaining_epochs = args.num_epoch - (epoch + 1)
        eta = format_duration(remaining_epochs * avg_epoch)
        epochs_pbar.set_postfix(train_acc=f"{train_acc:.4f}", val_acc=f"{val_acc:.4f}", eta=eta)

        print(
            f"Epoch {epoch + 1}/{args.num_epoch} | "
            f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"best_acc {best_acc:.4f} | eta {eta}"
        )

    print(
        f"Done | best_epoch {best_epoch} | best_acc {best_acc:.4f} | "
        f"best_f1 {best_f1:.4f} | best_acc_f1 {best_acc_f1:.4f}"
    )
    writer.close()


    eval_metrics = {'Starttime': starttime, 
                    'fold': args.n_fold,
                    'num_classes': args.num_classes,
                    'model': args.net,
                    'batch_size': args.batch_size,
                    'lr': args.lr,
                    'num_epoch': args.num_epoch,
                    'best_epoch': best_epoch,
                    'best_acc': best_acc,
                    'best_f1': best_f1,
                    'best_acc_f1': best_acc_f1,
                      }
    json_str = json.dumps(eval_metrics, indent=1)
    with open(os.path.join(args.result_dir, starttime[:10] +'_'+ args.json_dir), 'a') as f:
        f.write(json_str)
