##############################################
#    Switching Linear Dynamical System 
#    Code for both SLDS generative model as well
#    as variational inference code
##############################################
import torch 
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
import torch.nn.functional as F


class SLDS(nn.Module):
    def __init__(self, emb_size, hidden_size, embeddings=None, cell_type="GRU", layers=1, bidir=True, use_cuda=True):
            """
            For NOW, Only supporting GRU with 1 layer
            Args:
                emb_size (int) : size of input embeddings
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
                self.rnn = nn.LSTM(self.emb_size, self.hidden_size, self.layers, bidirectional=self.bidir, batch_first=True)
            else:
                self.rnn = nn.GRU(self.emb_size, self.hidden_size, self.layers, bidirectional=self.bidir, batch_first=True)

    def forward(input, hidden):
        raise NotImplementedError

    def initHidden(self, batch_size):
        dirs = 2 if self.bidir else 1
        if self.cell_type == "LSTM":
            hidden = (Variable(torch.zeros(batch_size, self.layers*dirs, self.hidden_size)),
                    Variable(torch.zeros(self.layers*dirs, batch_size, self.hidden_size)))
        else:
            hidden = Variable(torch.zeros(self.layers*dirs, batch_size, self.hidden_size))

        if self.use_cuda:
            return hidden.cuda()
        else:
            return hidden

class Encoder(EncDecBase):
    'Encoder class to use with the DVae'
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
            last state (Tuple(Tensor)) : (h_n, c_n) [batch, layers*directions, hidden_size]
        """
        out = self.embeddings(input).view(input.shape[0], input.shape[1], -1) #[batch, seq_len, emb_size]
        
        if use_packed:
            packed_input = pack_padded_sequence(out, seq_lens.cpu().numpy(), batch_first=True)
            self.rnn.flatten_parameters()
            packed_out, hidden = self.rnn(packed_input, hidden)
            enc_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            enc_out, hidden = self.rnn(out, hidden)

        return enc_out, hidden


class Decoder(EncDecBase):
    'Decoder (InputFeed) class to use with the DVae'

    def __init__(self, emb_size, hidden_size, embeddings=None, cell_type="GRU", layers=1, attn_dim=-1, use_cuda=True, dropout=0.0):

        #All the same except for attn_dim, which is the output of dim of attn, should be same as hidden_size 
        #attn_dim is tuple of (attnetion memory dim, attnention output dim)
        if attn_dim is None:
            attn_mem_dim = 2*hidden_size
            attndim = hidden_size
        else:
            attn_mem_dim, attndim = attn_dim

        bidir = False

        super(Decoder, self).__init__(emb_size + attndim, hidden_size, embeddings, cell_type, layers, bidir, use_cuda) 
        self.attn_dim = attndim
        #Previous output of attention, concat to input on text step, init to zero
        self.input_feed = None #Variable(torch.zeros(batch_size, self.attn_dim)) 
        self.attention = Attention((hidden_size, attn_mem_dim, self.attn_dim), use_cuda=self.use_cuda)
        
        if dropout > 0:
            print("Using a Dropout Value of {} in the decoder".format(dropout))
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = None

    def reset_feed_(self):
        """
        Reset the input feed so pytorch clears the memory
        """
        del self.input_feed
        self.input_feed = None

    def init_feed_(self, feed):
        """
        Initialize the input feed (usually with zero vector)
        Args
            feed (Tensor [batch_size, self.attn_dim)
        """
        if self.input_feed is None:
            self.input_feed = feed

    def forward(self, input, hidden, memory):
        """
        Run a SINGLE computation step
        Concat the inputfeed to the input,
        Update the RNN state
        Then compute the attention vector output
        Args:
            input (Tensor, [batch]) : Tensor of input ids for the embeddings lookup (one per batch since its done one at a time)
            hidden (Tuple(FloatTensor)) : (h, c) if LSTM, else just h, previous state
            memory (Tensor, [batch, num_latents, latent_dim]) : The latents to attend over (we have num_latents of them)

        Returns:
            output (Tensor, [batch, attn_dim]) : output at all time steps 
                   (the output is the output from attention (the pre logits))
            last state (Tuple(Tensor)) : (h_n, c_n) [batch, layers, hidden_size], the actual last state
        """
        if self.drop is None:
            out = self.embeddings(input).view(input.shape[0], -1) #[batch, emb_size]
        else:
            out = self.drop(self.embeddings(input).view(input.shape[0], -1)) #[batch, emb_size]

        #concat input feed 
        dec_input = torch.cat([out, self.input_feed], dim=1).unsqueeze(dim=1) #[batch, emb_size + attn_dim]
        self.rnn.flatten_parameters()
        rnn_output, hidden = self.rnn(dec_input, hidden) #rnn_output is hidden state of last layer
        #rnn_output dim is [batch, 1, hidden_size]
        rnn_output=torch.squeeze(rnn_output, dim=1)
        dec_output, scores = self.attention(rnn_output, memory)
        if self.drop is not None:
            dec_output = self.drop(dec_output)
        self.input_feed = dec_output #UPDATE Input Feed
        
       
        return dec_output, hidden

