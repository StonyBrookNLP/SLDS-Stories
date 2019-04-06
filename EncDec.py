#Classes for basic encoder/decoder stuff
import torch 
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncDecBase(nn.Module):
    def __init__(self, emb_size, hidden_size, embeddings=None, cell_type="GRU", layers=2, bidir=True, use_cuda=True):
            """
            Args:
                emb_size (int) : size of inputs to rnn
                hidden_size (int) : size of hidden 
                embeddings (nn.Module) : Torch module (with same type signatures as nn.Embeddings) to use for embedding
                cell_type : LSTM or GRU
                bidir (bool) : Use bidirectional encoder?
            """
            super(EncDecBase, self).__init__()
            self.emb_size = emb_size
            self.hidden_size = hidden_size
            self.embeddings = embeddings
            self.layers = layers
            self.bidir = bidir
            self.cell_type = cell_type
            self.use_cuda = use_cuda
            if cell_type == "LSTM":
                self.rnn = nn.LSTM(self.emb_size, self.hidden_size, self.layers, bidirectional=self.bidir)
            else:
                self.rnn = nn.GRU(self.emb_size, self.hidden_size, self.layers, bidirectional=self.bidir)

    def forward(input, hidden):
        raise NotImplementedError

    def initHidden(self, batch_size):
        dirs = 2 if self.bidir else 1
        if self.cell_type == "LSTM":
            hidden = (Variable(torch.zeros(self.layers*dirs, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.layers*dirs, batch_size, self.hidden_size)))
        else:
            hidden = Variable(torch.zeros(self.layers*dirs, batch_size, self.hidden_size))

        if self.use_cuda:
            return hidden.cuda()
        else:
            return hidden

class Encoder(EncDecBase):
    def forward(self, input, hidden, seq_lens, use_packed=True):
        """
        Encode an entire sequence
        Args:
            input (Tensor, [batch, seq_len]) : Tensor of input ids for the embeddings lookup
            seq_lens (Tensor [seq_len]) : Store the sequence lengths for each batch for packing
            hidden (Tuple(FloatTensor)) : (h, c) if LSTM, else just h
            use_packed (bool) : whether to use a packed sequence

        Returns:
            output (Tensor, [batch, seq_len, hidden*directions) : output at each time step
            last state (Tuple(Tensor)) : (h_n, c_n) [layers*directions, batch, hidden_size]
        """
        out = self.embeddings(input).transpose(0,1) #[seq_len, batch, emb_size]
        
        if use_packed:
            packed_input = pack_padded_sequence(out, seq_lens.cpu().numpy())
            self.rnn.flatten_parameters()
            packed_out, hidden = self.rnn(packed_input, hidden)
            enc_out, _ = pad_packed_sequence(packed_out)
        else:
            #self.rnn.flatten_parameters()
            enc_out, hidden = self.rnn(out, hidden)

        return enc_out.transpose(0,1), hidden #return enc_out as batch, seq_len, hidden_dim


class Decoder(EncDecBase):

    def __init__(self, emb_size, hidden_size, embeddings=None, cell_type="GRU", layers=1, use_cuda=False, dropout=0.0):
        bidir=False
        super(Decoder, self).__init__(emb_size, hidden_size, embeddings, cell_type, layers, bidir, use_cuda) 
        
        if dropout > 0:
            print("Using a Dropout Value of {} in the decoder".format(dropout))
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = None

    def forward(self, input, hidden, concat=None):
        """
        Run a SINGLE computation step
        Update the RNN state
        Args:
            input (Tensor, [batch]) : Tensor of input ids for the embeddings lookup (one per batch since its done one at a time)
            hidden (Tuple(FloatTensor)) : (h, c) if LSTM, else just h, previous state
            concat (Tensor, [batch, hidden_size]) : optional item to concatenate to the input at this time step

        Returns:
            output (Tensor, [batch, hidden_dim]) : output for current time step (hidden state of last layer)
                   (the output is the output from attention (the pre logits))
           hidden(Tuple(Tensor)) : (h_n, c_n) [layers, batch, hidden_size], the actual last state
        """
        if self.drop is None:
            out = self.embeddings(input).unsqueeze(dim=0) #[seq_len=1, batch, emb_size]
        else:
            out = self.drop(self.embeddings(input).unsqueeze(dim=0))

        if concat is None:
            dec_input = out
        else:
            dec_input = torch.cat([out.squeeze(0), concat], dim=1).unsqueeze(dim=0) #[1, batch, emb_size + hidden_size]

        #self.rnn.flatten_parameters()
        rnn_output, hidden = self.rnn(dec_input, hidden) #rnn_output is hidden state of last layer, hidden is for all layers (gets passed for next tstep)
        #rnn_output dim is [1, batch, hidden_size]
        rnn_output=torch.squeeze(rnn_output, dim=0)

        #if self.drop is not None:
        #    rnn_output = self.drop(rnn_output)
        
        return rnn_output, hidden


def gather_last(input, lengths, use_cuda=False):
    """
    In a padded batched sequence, gather the last 
    element for each according to lengths
    Args
        input (Tensor [batch, seq_len, *]) : padded input
        dim (int) dimension to gather on
        lengths (Tensor [batch]): lengths of each batch
    Returns 
        Tensor [batch, *]
    """
    #print("seq len type {}".format(type(lengths)))
    index_vect = torch.max(torch.LongTensor(lengths.shape).zero_(), lengths - 1).view(lengths.shape[0], 1,1) #convert len to index
    index_tensor = torch.LongTensor(input.shape[0], 1, input.shape[2]).zero_() + index_vect
    if use_cuda:
        return torch.gather(input, 1, Variable(index_tensor.cuda())).squeeze(dim=1)
    else:
        return torch.gather(input, 1, Variable(index_tensor)).squeeze(dim=1)

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    Taken From OpenNMT code
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

def fix_enc_hidden(h):
    # The encoder hidden is (layers*directions) x batch x dim.
    # We need to convert it to  layers x batch x (directions*dim).
    #THIS ASSUMES h COMES FROM A BIDERECTIONAL (directions=2) ENCODER
    h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
    return h

def kl_divergence(q, p=None, use_cuda=True):
    """
    Compute the kl divergence, KL[q(z) || p(z)]
    between two multinomials. Assume p to be uniform if none
    Args:
        q (Variable torch.Tensor [*, dim])
        p Variable torch.Tensor([1,dim])
    """
    dim = q.shape[1]
    if p is None:
        a =torch.zeros(1,dim) + 1.0/dim
        if use_cuda:
            p = Variable(a.cuda())
        else:
            p = Variable(a)

    return torch.sum(q*(torch.log(q)-torch.log(p)), dim=1)




