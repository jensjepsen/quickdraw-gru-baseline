import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import helpers

def apply_rnn(rnn,x,drawing_lens):
    x = pack_padded_sequence(x,lengths=drawing_lens,batch_first=True)

    x,_ = rnn(x)

    x, lens = pad_packed_sequence(x,batch_first=True)

    # take the last output of stroke_rnn for each drawing in batch
    # and remove dim of size 1
    lens = (drawing_lens.view(-1,1,1) - 1).expand(-1,1,x.size(2))
    x = x.gather(dim=1,index=lens).squeeze(1)
    return x

class Net(nn.Module):
    def __init__(self,num_classes,hidden_size,num_layers):
        super(Net,self).__init__()
        self.hidden_size = hidden_size

        self.stroke_rnn = nn.GRU(2,hidden_size,batch_first=True,num_layers=num_layers)
        
        self.output = nn.Sequential(
            nn.Linear(hidden_size,num_classes)
        )

    def forward(self,x,drawing_lens=None):
        batch_dim, drawing_dim, feature_dim = x.size()

        x = apply_rnn(self.stroke_rnn,x,drawing_lens)
        
        return self.output(x)
