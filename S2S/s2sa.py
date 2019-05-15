################################################
#
# Sequence-to-sequence model with Bilinear attention
#
################################################

import torch 
import torch.nn as nn
import numpy as np
import math
#from torch.autograd import Variable
from EncDec import Encoder, Decoder, Attention, fix_enc_hidden, gather_last
import torch.nn.functional as F
import data_utils
#import generate as ge
from data_utils import EOS_TOK, SOS_TOK, PAD_TOK
from utils import variable

class S2SWithA(nn.Module):

    def __init__(self, emb_size, enc_hsize, dec_hsize, inp_vocab, out_vocab, 
        cell_type="GRU", layers=1, bidir=False, use_cuda=True, dropout=0.10):
        """
        Args:
            emb_size (int) : size of input word embeddings
            hsize (int or tuple) : size of the hidden state (for one direction of encoder). If this is an integer then it is assumed
            to be the size for the encoder, and decoder is set the same. If a Tuple, then it should contain (encoder size, dec size)
            layers (int) : layers for encoder and decoder
            vocab (Vocab object)
            bidir (bool) : use bidirectional encoder?
            cell_type (str) : 'LSTM' or 'GRU'
            sos_idx (int) : id of the start of sentence token
        """
        super(S2SWithA, self).__init__()

        self.embd_size=emb_size
        self.inp_vocab = inp_vocab
        self.out_vocab = out_vocab
        self.inp_vocab_size = len(inp_vocab.stoi.keys())
        self.out_vocab_size = len(out_vocab.stoi.keys())
        self.cell_type = cell_type
        self.layers = layers
        self.bidir = bidir
        self.sos_idx = self.out_vocab.stoi[SOS_TOK] #TODO check these
        self.eos_idx = self.out_vocab.stoi[EOS_TOK]
        self.inp_pad_idx = self.inp_vocab.stoi[PAD_TOK]
        self.out_pad_idx = self.out_vocab.stoi[PAD_TOK]
        self.use_cuda = use_cuda
        self.enc_hsize = enc_hsize
        self.dec_hsize = dec_hsize        
        if self.bidir:
            print("Bidirectional ON")
            bidir = 2
        else:
            bidir = 1

        #print("Got here enc {} and dec {}".format(self.enc_hsize, self.dec_hsize))
        #print("inp_vocab_size {} and out_vocab_size {}".format(self.inp_vocab_size, self.out_vocab_size))

        in_embedding = nn.Embedding(self.inp_vocab_size, self.embd_size, padding_idx=self.inp_pad_idx)
        out_embedding = nn.Embedding(self.out_vocab_size, self.embd_size, padding_idx=self.out_pad_idx)        
        self.encoder = Encoder(self.embd_size, self.enc_hsize, in_embedding, self.cell_type, self.layers, self.bidir, use_cuda=use_cuda)
        
        self.decoder = Decoder(self.embd_size, self.dec_hsize, out_embedding, self.cell_type, self.layers, attn_dim=(self.enc_hsize*bidir, self.dec_hsize), use_cuda=use_cuda, dropout=dropout)

        #self.bridge = nn.Linear(bidir*layers*self.enc_hsize, layers*self.dec_hsize)

        self.logits_out= nn.Linear(self.dec_hsize, self.out_vocab_size) #Weights to calculate logits, out [batch, vocab_size]
               

    def forward(self, input, seq_lens, output, beam_size=-1, str_out=False, max_len_decode=50, min_len_decode=0, n_best=1, encode_only=False):
        """
        Args
            input (Tensor, [batch, seq_len]) : Tensor of input ids for the embeddings lookup
            seq_lens (Tensor [seq_len]) : Store the sequence lengths for each batch for packing
            str_output (bool) : set to true if you want the textual (greedy decoding) output, for evaultation
                use a batch size of 1 if doing this
            encode_only (bool) : Just encoder the input and return dhidden (initial decoder hidden) and latents
        Returns
            : List of outputs of the decoder (the softmax distributions) at each time step 
        """

        batch_size = input.size(0) 

        if str_out: #use batch size 1 if trying to get actual output
            assert batch_size == 1

        # INIT THE ENCODER
        ehidden = self.encoder.initHidden(batch_size)

        #Encode the entire sequence
        #Output gives each state output, ehidden is the final state
        #output (Tensor, [batch, seq_len, hidden*directions] 
        #ehidden [batch, layers*directions, hidden_size]
        
        # RUN FORWARD PASS THROUGH THE ENCODER 
        enc_output, ehidden = self.encoder(input, ehidden, seq_lens)  
        #print("input {}, ehidden {}, seq_lens {}, enc_output {}".format(input.size(), ehidden.size(), seq_lens.size(), enc_output.size()))
        
        # [layers*bidir, batch_size, hidden_size] 
        dhidden = ehidden #self.bridge(ehidden.transpose(0, 1).contiguous().view(batch_size, -1)).view(-1, self.layers, self.dec_hsize).transpose(0,1)
        #print("dhidden size {} type {}".format(dhidden.size(), type(dhidden)))

        #Decode output one step at a time
        #THIS IS FOR TESTING/EVALUATING, FEED IN TOP OUTPUT AS INPUT (GREEDY DECODE)
        if str_out:
            if beam_size >= 0:
                print("BEAM DECODING NOT IMPLEMENTED")
                exit(0)

            # GREEDY Decoding 
            self.decoder.init_feed_(variable(torch.zeros(batch_size, self.decoder.attn_dim))) #initialize the input feed 0
           
            return self.greedy_decode(input, dhidden, enc_output, max_len_decode)


        # This is for TRAINING, use teacher forcing  
        self.decoder.init_feed_(variable(torch.zeros(batch_size, self.decoder.attn_dim))) #initialize the input feed 0 

        # Remember: input should be the output here
        return self.do_train(output, batch_size, dhidden, enc_output)


    def do_train(self, input, batch_size, dhidden, enc_output, return_hid=False, use_eos=True):
        #print("do_train input {}".format(input.size()))
        dec_outputs = []
 
        input_size = input.size(1)
        #print(f"input size {input_size}")       
        for i in range(input_size): # not processing EOS
            # Choose input for this step
            if i == 0:
                tens = torch.LongTensor(input.shape[0]).zero_() + self.sos_idx   
                dec_input = variable(tens) #Decoder input init with sos 
            else:  
                dec_input = input[:, i-1]
            #print(f"i {i} dec input {dec_input}")
     
            dec_output, dhidden = self.decoder(dec_input, dhidden, enc_output) 
            dec_outputs += [dec_output]
 
        dec_outputs = torch.stack(dec_outputs, dim=0) #DIMENSION SHOULD BE [Seq x batch x dec_hidden]
        if return_hid:
            return dhidden, dec_outputs 
        else:
            self.decoder.reset_feed_() #reset input feed so pytorch correctly cleans graph
        
        # logits is [seq_len * batch_size * vocab_size]
        logits = self.logits_out(dec_outputs) 
        logits = logits.transpose(0,1).contiguous() # convert to [batch, seq, vocab]
        #print("logits {}".format(logits.size())) 

        return logits


    def greedy_decode(self, input, dhidden, enc_output, max_len_decode):
        """
        assumption: input is 1 sentence. max_len_decode is the actual target length.
        """

        outputs = []
        dec_input = variable(torch.LongTensor(input.shape[0]).zero_() + self.sos_idx) 
        prev_output = variable(torch.LongTensor(input.shape[0]).zero_() + self.sos_idx)

        if self.decoder.input_feed is None: 
            self.decoder.init_feed_(variable(torch.zeros(1, self.decoder.attn_dim))) #initialize the input feed 0 

        
        for i in range(max_len_decode):
            if i == 0: #first input is SOS (start of sentence)
                dec_input = variable(torch.LongTensor(input.shape[0]).zero_() + self.sos_idx)
            else:
                dec_input = prev_output
            #print("STEP {} INPUT: {}".format(i, dec_input.data))

            dec_output, dhidden = self.decoder(dec_input, dhidden, enc_output)
            logits = self.logits_out(dec_output)

            probs = F.log_softmax(logits, dim=1)
            _, top_inds = probs.topk(1, dim=1)
            
            outputs.append(top_inds.squeeze().item())
            prev_output = top_inds

            if top_inds.squeeze().item() == self.eos_idx:
                break

        self.decoder.reset_feed_() #reset input feed so pytorch correctly cleans graph
        return outputs
  
