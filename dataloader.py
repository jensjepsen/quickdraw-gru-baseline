import torch
from torch.utils.data import Dataset,DataLoader
from glob import glob
import os
import zarr
from zarr import blosc
import csv
from collections import defaultdict, Counter
import json
import numpy as np
from tqdm import tqdm
import numcodecs
from numcodecs import Blosc
import pickle
import math
import helpers

try:
    import visdom
    vis = visdom.Visdom()
except ImportError:
    vis = None

def collate_flat(data):
    drawings, labels = zip(*data)
    drawings = drawings[0]
    labels = labels[0]
    drawings = [d for d in drawings]
    num_strokes = [len(drawing) for drawing in drawings]
    max_strokes = max(num_strokes)
    stroke_lens = [[len(s[0]) for s in d] for d in drawings]
    max_stroke_length = max(max(s) for s in stroke_lens)
    batch_size = len(drawings)
    drawings_tensor = np.zeros((batch_size,max_strokes * max_stroke_length,2),dtype=np.float32)
    drawing_lens = []
    for i, drawing in enumerate(drawings):
        strokes = []
        for j, stroke in enumerate(drawing):
            strokes.extend(zip(stroke[0],stroke[1]))
        drawing_lens.append(len(strokes))
        drawings_tensor[i,:len(strokes),:] = strokes
    
    # sort drawings in descending order of number of strokes
    # no need to save argsort result, as we don't need to restore original order 
    drawing_lens = np.array(drawing_lens)
    drawing_order = np.argsort(drawing_lens)[::-1]
    drawings_tensor = drawings_tensor[drawing_order]
    drawing_lens = drawing_lens[drawing_order]

    labels = np.array(labels)[drawing_order]
                
    return (torch.tensor(drawings_tensor),
            torch.tensor(labels).long(),
            torch.tensor(drawing_lens).long(),
            )

class QuickDrawDataset(Dataset):
    def __init__(
                self,max_strokes,max_stroke_length,
                batch_size,max_per_class=1000,
                root_dir=os.environ["QUICKDRAW_DATA_ROOT"],
                arr_dir="processed_data",transform=None):
        self.arr_dir = arr_dir
        self.root_dir = root_dir
        self.max_strokes = max_strokes
        self.max_stroke_length = max_stroke_length
        self.max_per_class = max_per_class
        self.batch_size = batch_size
        self.transform = transform

        self.zarr_kwargs = dict(
            compressor=Blosc(),
            chunks=(512,),
            dtype=object,
            object_codec=numcodecs.Pickle()
            )

        if not os.path.exists(self.get_arr_dir()):
            self.preprocess(root_dir)
       
        self.drawings, self.classes = (
            zarr.open(self.get_arr_path("drawings"),"r"),
            zarr.open(self.get_arr_path("classes"),"r")[:]
            )

        with open(self.get_json_path()) as f:
            d = json.load(f)
            self.class2label, self.country2label = d["class2label"], d["country2label"]

    def __len__(self):
        return  len(self.drawings) / self.batch_size

    def __getitem__(self,idx):
        idx = idx * self.batch_size
        offset = idx + self.batch_size

        idxs = np.arange(idx,offset)
        drawings = self.drawings[idx:offset]
        labels = [self.classes[i] for i in idxs]

        if self.transform:
            drawings = self.transform(drawings)

        return (
            self.drawings[idx:offset],
            labels
            )

    def get_dataloader(self,**kwargs):
        return DataLoader(self,**kwargs)

    def get_arr_dir(self):
        return os.path.join(self.root_dir,self.arr_dir)

    def get_arr_path(self,name):
        return os.path.join(self.get_arr_dir(),name) + ".zarr"

    def get_json_path(self):
        return os.path.join(self.get_arr_dir(),"meta") + ".json"


    def preprocess(self,root_dir):
        paths = glob(os.path.join(root_dir,"*.csv"))

        if not os.path.exists(self.get_arr_dir()):
            os.mkdir(self.get_arr_dir())

        stroke_count = defaultdict(lambda: 0)
        stroke_length = defaultdict(lambda: 0)
        contry_codes = set()
        classes = set()
        class_count = defaultdict(lambda: 0)

        def count(line):
            code, drawing, id, recognized, timestamp, cls = line
            drawing = json.loads(drawing)
            
            stroke_count[len(drawing)] += 1
            for stroke in drawing:
                stroke_length[len(stroke[0])] += 1
            contry_codes.add(code)
            classes.add(cls)
            class_count[cls] += 1
        
        print "Counting.."
        for path in tqdm(paths):
            with open(path,"rb") as f:
                reader = csv.reader(f)
                next(reader,None)
                for i, l in enumerate(reader):
                    if i >= self.max_per_class: break
                    count(l)
        if vis:
           stroke_lengths_arr = zip(*stroke_length.items())
           vis.bar(opts=dict(title="Stroke lengths"),X=stroke_lengths_arr[1],Y=stroke_lengths_arr[0])
           stroke_counts_arr = zip(*stroke_count.items())
           vis.bar(opts=dict(title="Stroke counts"),X=stroke_counts_arr[1],Y=stroke_counts_arr[0])
           class_count_arr = zip(*class_count.iteritems())
           vis.bar(X=class_count_arr[1],opts=dict(legend=class_count_arr[0]))
           vis.text("Classes: " + ", ".join(classes) + "<br />Contry codes: " + ", ".join(contry_codes))

        self.class2label = dict((k,i) for i,k in enumerate(classes))
        self.country2label = dict((k,i) for i,k in enumerate(contry_codes))

        with open(self.get_json_path(),"w") as f:
            json.dump(
                dict(
                    class2label=self.class2label,
                    country2label=self.country2label)
                ,f)

        total_drawings = sum(class_count.itervalues())

        drawings_zarr = zarr.open(self.get_arr_path("drawings"),shape=(total_drawings,),mode="w",**self.zarr_kwargs)
        classes_zarr = zarr.open(self.get_arr_path("classes"),chunks=(total_drawings,),mode="w",shape=(total_drawings),dtype=np.int32)

        files = [open(path,"rb") for path in paths]
        readers = [csv.reader(f) for f in files]
        for reader in readers:
            next(reader)

        counter = defaultdict(lambda: 0)

        def get_next():
            if not len(readers):
                return None
            reader = readers[np.random.randint(0,len(readers))]
            l = next(reader,None)
            r = helpers.process(l,self.class2label,self.max_strokes,self.max_stroke_length) if not l is None else None
            label = r[1]
            counter[label] += 1
            if l is None or counter[label] > self.max_per_class:
                readers.remove(reader)
                return get_next()
            else:
                return r
        
        zarr_idx = 0
        def save_batch(zarr_idx,dc):
            d, c = zip(*dc)
            drawings_zarr[zarr_idx:zarr_idx + len(dc)] = d
            classes_zarr[zarr_idx:zarr_idx + len(c)] = c
            
            zarr_idx += len(dc)
            return zarr_idx

        dc = []
        with tqdm(total=total_drawings) as pbar:
            while len(readers):
                r = get_next()
                if r is None: break
                dc.append(r)
                
                pbar.update(1)
                if len(dc) > self.max_per_class:
                    zarr_idx = save_batch(zarr_idx,dc)
                    dc = []
        if len(dc): save_batch(zarr_idx,dc)
