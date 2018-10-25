from PIL import Image, ImageDraw
import itertools
import numpy as np
from visdom import Visdom
import json
import torch
try:
    from visdom import Visdom
    vis = Visdom()
except ImportError:
    vis = None

def process(data,class2label,max_strokes,max_stroke_length,test=False):
    line = data
    
    if not test:
        code, drawing, id, recognized, timestamp, cls = line
    else:
        id, code, drawing = line
    
    drawing = json.loads(drawing)
    _stroke_count = len(drawing)
    if _stroke_count > max_strokes:
        drawing = drawing[:max_strokes]
    
    drawing_new = []
    for i,stroke in enumerate(drawing):
        _stroke_length = len(stroke[0])
        if _stroke_length > max_stroke_length:
            stroke = [s[:max_stroke_length] for s in stroke]
            _stroke_length = max_stroke_length
        drawing_new.append([[j / 255.0 for j in s] for s in stroke])
    
    if not test:
        label = class2label[cls]
        return drawing_new,label
    else:
        return (drawing_new,id)

def apk(actual, predicted, k=10):
    """
    THIS FUNCTION IS TAKEN FROM
    SOURCE: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def flatten(drawing):
	x = (f for stroke in drawing for f in stroke[0])
	y = (f for stroke in drawing for f in stroke[1])
	coords = zip(x,y)
	return coords

def stroke2coords(stroke):
	return zip(stroke[0],stroke[1])

def flip(drawing,max=255,dim=0):
        d = []
        for stroke in drawing:
            new_stroke = []
            for i,s in enumerate(stroke):
                 new_val = lambda j: max - j if dim == i else j
                 new_stroke.append([new_val(j) for j in s])
            d.append(new_stroke)
        return d

def noise(drawing,dev):
    d = []
    for stroke in drawing:
        new_stroke = []
        for i,s in enumerate(stroke):
             new_stroke.append([j + (np.random.rand(1)[0] - 0.5) * dev for j in s])
        d.append(new_stroke)
    return d

def flip_arr(drawings,max=1.0):
	mask = (drawings[:,:,:,0] != 0.0).float()
        #new_val = (torch.full_like(drawings,max) - drawings[:,:,:,0]) * mask
	drawings[:,:,:,0] = (max - drawings[:,:,:,0]) * mask
	return drawings

def noise_arr(drawings,max=255):
	mask = (drawings != 0.0) * 1.0
	drawings[:,:,:,0] += (np.random.rand(drawings.shape) - 0.5) * dev * mask
	return drawings

def draw(drawing,size=(255,255)):
	im = Image.new("L",size,color=0)
	draw = ImageDraw.Draw(im)
	
	for i in xrange(1,len(coords) - 1):
		coords = stroke2coords(drawing)
		draw.line(coords,fill=255)

	return im

def draw_arr(drawing,size=(255,255)):
	im = Image.new("L",size,color=0)
	draw = ImageDraw.Draw(im)

	for i in xrange(1,len(coords) - 1):
		coords = stroke2coords(drawing)
		coords = [(x,y) for x,y in coords if (x != 0.0 or y != 0.0)]
		draw.line(coords,fill=255)

	return im

def plot(drawing):
	im = np.array(draw(drawing))
	vis.image(im)
