# A simple GRU baseline for the QuickDraw Kaggle competition
A GRU RNN followed by a fully connected layer, for the [QuickDraw Kaggle competition](https://www.kaggle.com/c/quickdraw-doodle-recognition).

## Training
First, download the training data, as found on Kaggle.com. The dataloader expects an environment variable `QUICKDRAW_DATA_ROOT` to point to a directory containing the simplified csv files.

Then run `train.py` to train the model. The first time this is run, the csv files will be preprocessed and stored in a zarr array in the subfolder `processed_data` in the folder containing the QuickDraw csv files.

Reading from the zarr array is fastest if done in chunks, so the data from the csv files are read in a round-robin fashion and added to the array, so that each class is roughly uniformly distributed between chunks. When the batches are read from the array during training, they are also read in chunks, which means that a batch always contains the same data points. This doesn't seem to adversely affect convergence or generalization, but speeds up data loading significantly.

The loss is categorical cross entropy and the optimizer is Adam with default parameters.

### Options
```
python train.py --help

usage: train.py [-h] [--mode {train,load}] [--batch_size BATCH_SIZE]
                [--hidden_size HIDDEN_SIZE] [--learning_rate LEARNING_RATE]
                [--clip_grad_norm CLIP_GRAD_NORM] [--epochs EPOCHS]
                [--max_per_class MAX_PER_CLASS] [--max_strokes MAX_STROKES]
                [--max_stroke_length MAX_STROKE_LENGTH]
                [--num_layers NUM_LAYERS]

optional arguments:
  -h, --help            show this help message and exit
  --mode {train,load}
  --batch_size BATCH_SIZE
  --hidden_size HIDDEN_SIZE
                        Hidden size of all layers of the model
  --learning_rate LEARNING_RATE
                        Learning rate for the Adam optimizer
  --clip_grad_norm CLIP_GRAD_NORM
  --epochs EPOCHS
  --max_per_class MAX_PER_CLASS
                        Number of examples to read from csv files for each
                        class
  --max_strokes MAX_STROKES
                        Max strokes per drawing, drawings with more strokes
                        will be truncated
  --max_stroke_length MAX_STROKE_LENGTH
                        Max points per stroke, strokes with more points will
                        be truncated
  --num_layers NUM_LAYERS
                        Number of GRU layers
```

## Requirements
- Python 2.7
- PyTorch
- NumPy
- tqdm
- Pillow (or PIL)
- Visdom (for visualisation during preprocessing)

