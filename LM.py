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
import itertools
import torch.nn.functional as F
import utils
from masked_cross_entropy import masked_cross_entropy
from EncDec import Encoder, Decoder, gather_last, sequence_mask
from data_utils import EOS_TOK, SOS_TOK, PAD_TOK, transform


class LM(nn.Module):
    def __init__(self, hidden_size, rnn_hidden_size, embd_size, vocab, trans_matrix, layers=2, pretrained=False, dropout=0.0, use_cuda=False):
            """
            Args:
                hidden_size (int) : size of hidden vector z 
                embd_size (int) : size of word embeddings
                vocab (torchtext.Vocab) : vocabulary object
                trans_matrix (Tensor, [num states, num states]) : Transition matrix probs for switching markov chain
                pretrained (bool) : use pretrained word embeddings?

            """
            super(LM, self).__init__()
            #self.hidden_size = self.dec_hsize = hidden_size
            #self.encoded_data_size = hidden_size #Vector size to use whenever encoding text into a %&#ing vector
            self.hidden_size = hidden_size
            self.encoded_data_size = self.dec_hsize = rnn_hidden_size #Vector size to use whenever encoding text into a %&#ing vector

            self.trans_matrix = trans_matrix
            self.num_states = trans_matrix.shape[0]
            self.embd_size=embd_size
            self.layers = layers
            self.use_cuda = use_cuda 
            
            self.vocab_size=len(vocab.stoi.keys())
            self.sos_idx = vocab.stoi[SOS_TOK]
            self.eos_idx = vocab.stoi[EOS_TOK]
            self.pad_idx = vocab.stoi[PAD_TOK]

            in_embedding = nn.Embedding(self.vocab_size, self.embd_size, padding_idx=self.pad_idx)
            out_embedding = nn.Embedding(self.vocab_size, self.embd_size, padding_idx=self.pad_idx)

            if pretrained:
                print("Using Pretrained")
                in_embedding.weight.data = vocab.vectors
                out_embedding.weight.data = vocab.vectors


            self.liklihood_rnn= Decoder(self.embd_size, self.dec_hsize, out_embedding, "GRU", self.layers, use_cuda=use_cuda, dropout=dropout)
            self.liklihood_logits= nn.Linear(self.dec_hsize, self.vocab_size, bias=False) #Weights to calculate logits, out [batch, vocab_size]


            if use_cuda:
                self.liklihood_rnn = self.liklihood_rnn.cuda()
                self.liklihood_logits = self.liklihood_logits.cuda()

    def forward(self, input, seq_lens, gumbel_temp=1.0, state_labels=None):
        """
        Args
            input (Tensor, [num_sents, batch, seq_len]) : Tensor of input ids for the embeddings lookup
            seq_lens (Tensor [num_sents, batch]) : Store the sequence lengths for each batch for packing
            state_labels (tensor [num_sents, batch]) : labels for the states if doing supervision
        Returns
            output logits (Tensor, [num_sents, batch, seq_len, vocab size]) : logits for the output words
            state logits (Tensor, [num_sents, batch, num classes]) : logits for state prediction, can be used for supervision and to calc state KL
            Z_kl (Tensor, [batch]) : kl diveragence for the Z transitions (calculated for each batch)
        """
        #Now Evaluate the Liklihood for each sentence
        batch_size = input.size(1)
        num_sents = input.size(0)

        dhidden=None
        data_logits = []
        for i in range(num_sents):
            logits, dhidden = self.data_liklihood_factor(input[i,:,:], dhidden, seq_lens[i]) # USE IF PASSING PREV HIDDEN STATE TO NEXT
            data_logits.append(logits)

        data_logits = torch.stack(data_logits, dim=0) #[num_sents, batch, seq, num classes]

        return data_logits

    #P(X | Z)
    def data_liklihood_factor(self, data, dhidden=None, lengths=None):
        """
        Output the logits at each timestep (the data liklihood outputs from the rnn)
        Args:
            data (Tensor, [batch, seq_len]) vocab ids of the data
            curr_z (Tensor, [batch, hidden_size]) 
        Ret:
            logits (Tensor, [batch, seq_len, vocab size])
        """
        #REMEMBER: The data has both BOS and EOS appended to it

        dhidden_list = [] #List for storing dhiddens so that the last one before pads can be extracted out 

        if dhidden is None:
            dhidden = torch.zeros(self.layers, data.shape[0], self.dec_hsize).cuda() if self.use_cuda else torch.zeros(self.layers, data.shape[0], self.dec_hsize)

        logits = []
        for i in range(data.size(1)-1): #dont process last (the eos)

            dec_input = data[:, i]

            #dec_output is [batch, hidden_dim]
            dec_output, dhidden = self.liklihood_rnn(dec_input, dhidden) 
            logits_t = self.liklihood_logits(dec_output)
            logits += [logits_t]

            dhidden_list += [dhidden.transpose(0,1)] #list stores [batch, layers, hiddensize]
 
        logits = torch.stack(logits, dim=1) #DIMENSION BE [batch x seq x num_classes]

        dhidden_list = torch.stack(dhidden_list, dim=1) #[batch x seq x layers x hidden_size]
        dhidden_list = dhidden_list.view(dhidden_list.shape[0], dhidden_list.shape[1], -1) #[batch, seq, layers*hidden_size]
        last_dhidden = gather_last(dhidden_list, lengths - 1, use_cuda=self.use_cuda)
        last_dhidden = last_dhidden.view(-1, self.layers, self.dec_hsize).transpose(0,1).contiguous()  #[layers, batch, hiddensize]

        return logits, last_dhidden

    def greedy_decode(self, dhidden=None, max_decode=30, top_k=15):
        """
        Output the logits at each timestep (the data liklihood outputs from the rnn)
        Args:
            data (Tensor, [batch, seq_len]) vocab ids of the data
            curr_z (Tensor, [batch, hidden_size]) 
        Ret:
            outputs - list of indicies
        """

        if dhidden is None:
            dhidden = torch.zeros(self.layers, 1, self.dec_hsize)

        outputs = []
        prev_output = Variable(torch.LongTensor(1).zero_() + self.sos_idx)

        for i in range(max_decode): 
            dec_input = prev_output

            dec_output, dhidden = self.liklihood_rnn(dec_input, dhidden) 
            logits_t = self.liklihood_logits(dec_output)

            #dec_output is [batch, hidden_dim]

           # probs = F.log_softmax(logits_t, dim=1)
           # top_vals, top_inds = probs.topk(1, dim=1)

            logits_t = self.top_k_logits(logits_t, k=top_k) 
            probs = F.softmax(logits_t/1.00, dim=1)
            top_inds = torch.multinomial(probs, 1)

            outputs.append(top_inds.squeeze().item())
            prev_output = top_inds[0]

            if top_inds.squeeze().item() == self.eos_idx:
                break

        return outputs, dhidden

    def top_k_logits(self,logits, k):
        vals,_ = torch.topk(logits,k)
        mins = vals[:,-1].unsqueeze(dim=1).expand_as(logits)
        return torch.where(logits < mins, torch.ones_like(logits)*-1e10,logits)


    def reconstruct(self, input, seq_lens, initial_sents):
        """
        Args
            input (Tensor, [num_sents, batch, seq_len]) : Tensor of input ids for the embeddings lookup
            seq_lens (Tensor [num_sents, batch]) : Store the sequence lengths for each batch for packing
        Returns
            output logits (Tensor, [num_sents, batch, seq_len, vocab size]) : logits for the output words
            state logits (Tensor, [num_sents, batch, num classes]) : logits for state prediction, can be used for supervision and to calc state KL
            Z_kl (Tensor, [batch]) : kl diveragence for the Z transitions (calculated for each batch)
        """
        batch_size = 1
        num_sents = input.size(0)

        dhidden=None
        outputs = []
        for i in range(num_sents):
            if i < initial_sents:
                _, dhidden = self.data_liklihood_factor(input[i, :, :], dhidden, seq_lens[i])
                sent_out = input[i, :, :].squeeze().tolist()
            else:
                sent_out, dhidden= self.greedy_decode(dhidden) 

            outputs.append(sent_out)

        return outputs

    def set_use_cuda(self, value):
        self.use_cuda = value
        self.liklihood_rnn.use_cuda = value

    def interpolate(self, input, seq_lens, initial_sents, num_samples, vocab):
        """
        Args
            input (Tensor, [num_sents, batch, seq_len]) : Tensor of input ids for the embeddings lookup
            seq_lens (Tensor [num_sents, batch]) : Store the sequence lengths for each batch for packing
        Returns
            output logits (Tensor, [num_sents, batch, seq_len, vocab size]) : logits for the output words
            state logits (Tensor, [num_sents, batch, num classes]) : logits for state prediction, can be used for supervision and to calc state KL
            Z_kl (Tensor, [batch]) : kl diveragence for the Z transitions (calculated for each batch)
        """
        batch_size = 1
        num_sents = input.size(0)
        min_cross_entropy = 10000.0
        best_outputs = None

        for _ in range(num_samples):
            dhidden=None
            outputs = []
            for i in range(num_sents):
                if i < initial_sents:
                    _, dhidden = self.data_liklihood_factor(input[i, :, :], dhidden, seq_lens[i]) #Just run the sentence through the lm so we can get the dhidden
                    sent_out = input[i, :, :].squeeze().tolist()
                elif i == num_sents-1:
                    last_logits, dhidden = self.data_liklihood_factor(input[i, :, :], dhidden, seq_lens[i]) #Just run the sentence through the lm so we can get the dhidden
                    sent_out = input[i, :, :].squeeze().tolist()
                else:   #Decode a new sentence
                    sent_out, dhidden= self.greedy_decode(dhidden) 

                outputs.append(sent_out)

            cross_entropy = masked_cross_entropy(last_logits, torch.LongTensor(outputs[-1][1:]).unsqueeze(dim=0), seq_lens[-1] - 1, use_cuda=self.use_cuda).item()
            if cross_entropy < min_cross_entropy:
                min_cross_entropy = cross_entropy
                best_outputs = outputs
                print("Cross Entropy: {}".format(min_cross_entropy))
                for j, sent in enumerate(best_outputs):
                    print("{}".format(transform(best_outputs[j], vocab.itos)))

                print("--------------------\n\n")


        return best_outputs

    def set_use_cuda(self, value):
        self.use_cuda = value
        self.liklihood_rnn.use_cuda = value



