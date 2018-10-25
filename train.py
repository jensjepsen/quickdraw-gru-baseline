import torch
import numpy as np
from torch import optim
from torch import nn
from dataloader import QuickDrawDataset, collate_flat
from torch.utils.data import DataLoader, Subset
from model import Net
from tqdm import tqdm
import helpers
import math

DEVICE = "cuda:0"

def get_outputs(model,args,kwargs,transforms):
    batch_out = None
    args = list(args)
    for transform in transforms:
        args[0] = transform(args[0])
        model_out = model(*args,**kwargs).to("cpu").numpy()
        if batch_out is None:
            batch_out = model_out.copy()
        else:
            batch_out += model_out
    return batch_out

def val(model,test):
    model.eval()
    visdom_windows = None
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        map3 = 0.0
        for i,b in enumerate(tqdm(test)):
            args, kwargs, labels = parse_batch(b)
            labels = labels.to("cpu").numpy()
            model_out = get_outputs(model,args,kwargs,[lambda x: x])
            correct += (model_out.argmax(axis=1) == labels).sum()
            top_3_out = model_out.argsort(axis=1)[:,::-1][:,:3]
            for j,label in enumerate(labels):
                map3 += float(helpers.apk([label],top_3_out[j],k=3))
            total += labels.shape[0]
        map3 = map3 / float(total)
        print "{}%, {}/{}, {}".format(correct / total,correct,total,map3)
    model.train()
    return correct / total

def parse_batch(b):
    drawings, labels, drawing_lens = b

    return (
        (drawings.to(DEVICE),),
        dict(drawing_lens=drawing_lens.to(DEVICE)),
        labels.to(DEVICE)
        )

def train(
        load, batch_size,
        hidden_size, learning_rate,
        clip_grad_norm, max_per_class,
        max_strokes, max_stroke_length,
        epochs, num_layers):
    if load:
        loaded = torch.load("model.pt")

    dataset_kwargs = dict(
        max_strokes=max_strokes,
        max_stroke_length=max_stroke_length,
        batch_size=batch_size,
        max_per_class=max_per_class,
        )

    dataset = QuickDrawDataset(**dataset_kwargs)

    if not load:
        idx = np.arange(len(dataset))
        np.random.shuffle(idx)
        split = int(math.floor(0.8 * len(dataset)))
        train_split, test_split = idx[:int(split)], idx[split:]
    else:
        train_split, test_split = loaded["idxs"]

    (train_set, test_set) = (
        Subset(dataset,train_split),
        Subset(dataset,test_split)
    )
    
    dataloader_kwargs = dict(num_workers=8, shuffle=True, batch_size=1, collate_fn=collate_flat)

    train = DataLoader(train_set,**dataloader_kwargs)
    test = DataLoader(test_set,**dataloader_kwargs)
    
    if not load:
        model = Net(num_classes=len(dataset.class2label),
                    hidden_size=hidden_size,
                    num_layers=num_layers)
    else:
        print "Notice: Using model loaded from model.pt"
        model = loaded["model"]

    model = model.to(DEVICE)

    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad),lr=learning_rate)

    nn.utils.clip_grad_norm_(model.parameters(),clip_grad_norm)

    criterion = nn.CrossEntropyLoss(size_average=True).to(DEVICE)
    
    def save(epoch,model):
        info = dict(
            model=model,
            idxs=(train_split,test_split),
            optimizer=optimizer.state_dict(),
            epoch=epoch,
        )
        torch.save(info,"model.pt")

    loss_mavg = 0.0
    epoch = 0
    if load:
        epoch = loaded.get("epoch",0)

    for i in xrange(epoch,epochs):
        model.train()

        for j,b in enumerate(tqdm(train)):
            args, kwargs, labels = parse_batch(b)
            optimizer.zero_grad()

            model_out = model(*args,**kwargs)
            loss = criterion(model_out,labels)
            loss.backward()
            optimizer.step()
            loss_mavg = loss_mavg * 0.9 + loss.item() * 0.1
            if j % 100 == 0:
                print "Epoch {}, Batch {}, Moving Loss {}, lr {}"\
                            .format(i,j,loss_mavg,[g["lr"] for g in optimizer.param_groups])

        val(model,test)
        save(epoch=i,model=model)

if __name__ == "__main__":
    import sys

    import argparse
    ap = argparse.ArgumentParser()
    
    ap.add_argument("--mode",default="train",choices=["train","load","test"])
    ap.add_argument("--batch_size",default=128,type=int)
    ap.add_argument("--hidden_size",default=128,type=int)
    ap.add_argument("--learning_rate",default=0.001,type=float)
    ap.add_argument("--clip_grad_norm",default=9.0,type=float)
    ap.add_argument("--epochs",default=1000,type=int)
    ap.add_argument("--max_per_class",default=1000,type=int)
    ap.add_argument("--max_strokes",default=30,type=int)
    ap.add_argument("--max_stroke_length",default=50,type=int)
    ap.add_argument("--num_layers",default=50,type=int)

    
    args = ap.parse_args()
    args = vars(args)
    mode = args.pop("mode")
    if mode == "test":
        test()
    else:
        train(mode=="load",**args)
        
