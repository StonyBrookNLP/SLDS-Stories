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
import utils
from EncDec import Encoder, Decoder, gather_last


class SLDS(nn.Module):
    def __init__(self, hidden_size, trans_matrix):
            """
            For NOW, Only supporting GRU with 1 layer
            Args:
                hidden_size (int) : size of hidden vector z 
                trans_matrix (Tensor, [num states, num states]) : Transition matrix probs for switching markov chain
            

            """
            super(SLDS, self).__init__()
            self.hidden_size = hidden_size
            self.encoded_data_size = 2*hidden_size #Vector size to use whenever encoding text into a %&#ing vector
            self.trans_matrix = trans_matrix
            self.num_states = trans_matrix.shape[0]

            in_embedding = nn.Embedding(self.vocab_size, self.embd_size, padding_idx=self.pad_idx)
            out_embedding = nn.Embedding(self.vocab_size, self.embd_size, padding_idx=self.pad_idx)

            self.dynamics_mean = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(self.num_states)])
            #self.dynamics_logvar = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(self.num_states)])
            self.dynamics_logvar = Variable(torch.randn([self.num_states, self.hidden_size])) #diagnol covariance [states, hidden] 


            self.sentence_encode_rnn= Encoder(self.embd_size, self.encoded_data_size / 2, in_embedding, self.cell_type, self.layers, bidir=True, use_cuda=use_cuda)
            self.liklihood_rnn= Decoder(self.embd_size, self.dec_hsize, out_embedding, self.cell_type, self.layers, use_cuda=use_cuda, dropout=dropout)
            self.liklihood_logits= nn.Linear(self.dec_hsize, self.vocab_size) #Weights to calculate logits, out [batch, vocab_size]

            #Variational Parameters
            self.dynamics_posterior_network = nn.Sequential(
                                                nn.Linear(self.hidden_size + self.encoded_data_size*2, self.hidden_size),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size, self.hidden_size)
                                              )
            self.state_posterior_network = nn.Sequential(
                                                nn.Linear(self.encoded_data_size*2, self.hidden_size),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size, self.num_states)
                                              )

    def forward(self, input, seq_lens):
        """
        Args
            input (Tensor, [num_sents, batch, seq_len]) : Tensor of input ids for the embeddings lookup
            seq_lens (Tensor [num_sents, batch]) : Store the sequence lengths for each batch for packing
        Returns
            output logits (Tensor, [batch, seq_len, vocab size]) : logits for the output words
            means
            state_logits
        """
        batch_size = input.size(1)
        num_sents = input.size(0)

        encoded_sents = self.encode_sentences(input, seq_lens) #[num_sents, batch, encoder dim)
        S_samples, state_logits = sample_switch_posterior(self, encoded_sents)
        Z_samples, Z_means, prior_means, logvars = sample_hidden_posterior(encoded_sents, S_samples) #[num_sents, batch, hidden_dim]
       
        #Now Evaluate the Liklihood for each sentence

        data_logits = []
        for i in range(num_sents):
            logits = self.data_liklihood_factor(input[i,:,:], Z_samples[i]) #logits is [batch, seq, num_classes]
            data_logits.append(logits)


        data_logits = torch.stack(data_logits, dim=0) #[num_sents, batch, seq, num classes]

        #ALSO NEED TO CALC THE KL DIVERGENCE, AVERAGE ACROSS BATCHES AND SENTENCES, IS CALCED AS
        #(Z_means - prior_means)^T exp(-logvars) (Z_means-prior_means)

        return data_logits, Z_means, state_logits

    #P(Z_i | Z_i-1, S_i)
    def dynamics_factor(self, prev_z, switch_state):
        """
        Output the distribution over Z_i given the prev Z and the current switching state
        Args:
            prev_z (Tensor, [batch, hidden_size]) 
            switch_state(Tensor, [batch, num_states]) : a probabilistic vector that sums to one over all states
        Ret:
            (mean, logvar)
            mean - Tensor [batch, hidden_size]
            logvar - Tensor [batch, hidden_size] (diagonal variance)
        """
        all_mean_weights = torch.stack([x.weight for x in self.dynamics_mean]) #[num_states X hidden X hidden]
        #all_var_weights = torch.stack([x.weight for x in self.dynamics_logvar])
        #crate single matrix as a convex comb of all dynamics matrix, using switch state as weights, 1 matrix for each batch
        mean_param = torch.einsum('ijk,bi->bjk', [all_mean_weights, switch_state])  #[batch X hidden X hidden]
        var_param = torch.einsum('ij,bi->bj', [self.dynamics_logvar, switch_state]) #[batch X hidden]

        means = torch.einsum('bij,bj->bi',[mean_param, prev_z]) #batch X hidden]
        return (means, var_param)

    
    #Variational Posterior Approximation
    #Q(Z_i | Z_i-1, S_i, X)
    def dynamics_posterior_factor(self, prev_z, switch_state, encoded_data):
        """
        Output the distribution over Z_i given the prev Z and the current switching state
        Learns the residual between the prior and the new data
        Args:
            prev_z (Tensor, [batch, hidden_size]) 
            switch_state(Tensor, [batch, num_states]) : a probabilistic vector that sums to one over all states
            encoded_data (Tensor, [batch, encoded_data_size]) - Vector representation of the data X (for example, from an LSTM encoder)
        Ret:
            (mean, logvar, means_prior)
            mean - Tensor [batch, hidden_size]
            logvar - Tensor [batch, hidden_size] (diagonal variance)
            means_prior - Same shape as means, the prior mean
        """
        means_prior, var_param = self.dynamics_factor(prev_z, switch_state):

        network_input = torch.cat((prev_z, encoded_data), dim=1) 
        mean_residual = self.dynamics_posterior_network(network_input)
        means = means_prior + mean_residual #Using the average might be another posibility

        return (means, var_param, means_prior)

    #Variational Posterior Approximation
    #Q(S_i | X)
    def state_posterior_factor(self, encoded_data):
        """
        Return the logits for the distribution over discrete switching states
        Args:
            encoded_data (Tensor, [batch, encoded_data_size]) - Vector representation of the data X (for example, from an LSTM encoder)
        Ret:
            logits - Tensor [batch, num classes]
        """
        return self.state_posterior_network(encoded_data)

    #P(X | Z)
    def data_liklihood_factor(self, data, curr_z):
        """
        Output the logits at each timestep (the data liklihood outputs from the rnn)
        Args:
            data (Tensor, [batch, seq_len]) vocab ids of the data
            curr_z (Tensor, [batch, hidden_size]) 
        Ret:
            logits (Tensor, [batch, seq_len, vocab size])
        """
        #INIT THE HIDDEN STATE WITH CURR_Z HERE
    
        logits = []
        for i in range(data.size(1) + 1): #PROCESS LAST
            #Choose input for this step
            if i == 0:
                tens = torch.LongTensor(data.shape[0]).zero_() + self.sos_idx
                if self.use_cuda:
                    dec_input = Variable(tens.cuda()) #Decoder input init with sos
                else:
                    dec_input = Variable(tens)
            else:  
                dec_input = data[:, i-1]

            #dec_output is [batch, hidden_dim]
            dec_output, dhidden = self.liklihood_rnn(dec_input, dhidden) 
            logits_t = self.liklihood_logits(dec_output)
            logits += [logits_t]
 
        logits = torch.stack(dec_outputs, dim=1) #DIMENSION BE [batch x seq x num_classes]
        return logits

    def encode_sentences(self, input, seq_lens):
        """
        Encode each sentence into vector representation
        Args
            input (Tensor, [num_sents, batch seq_len]) : Tensor of input ids for the embeddings lookup
            seq_lens (Tensor [num_sents, batch]) : Store the sequence lengths for each batch for packing
        Returns
            encoded sentences (Tensor, [num_sents, batch, vocab size]) : logits for the output words
        """

        num_sents = input.size(0)
        batch_size = input.size(1)

        encoded_sents = []
        ehidden = self.encoder.initHidden(batch_size)

        for i in range(num_sents):
            enc_output, ehidden = self.encoder(input[i,:,:], ehidden, seq_lens[i])
            last_output = gather_last(enc_output, seq_lens[i]) #[batch, encoder_dim]
            encoded_sents.append(last_output)

        encoded_sents = torch.stack(encoded_sents, dim=0) #[num_sents, batch, encoder dim)
        return encoded_sents


    def sample_switch_posterior(self, encoded_sents, gumbel_temp=1.0):
        """
        Sample the switching state posterior given the encoded data
        Args
            encoded_sents (Tensor, [num_sents, batch, encoded dim])  
        Returns
            state samples (Tensor, [num_sents, batch, num classes]) : samples from gumbel softmax for each step
            state logits (Tensor, [num_sents, batch, num classes]) : logits for state prediction, can be used for supervision
        """

        num_sents = encoded_sents.size(0)
        S_samples = []
        logit_list = []
        for i in range(num_sents):
            context_vect = utils.get_context_vector(encoded_sents, i, future=False)
            target_vect = encoded_sents[i, :, :]
            logits = self.state_posterior_factor(torch.cat([target_vect, context_vect], dim=1)) #[batch, num classes]
            sample = utils.gumbel_softmax_sample(logits, temp=gumbel_temp)
            S_samples.append(sample)
            logit_list.append(logits)

        S_samples = torch.stack(S_samples, dim=0) #[num_sents, batch, num classes]
        logit_list = torch.stack(logit_list, dim=0)
        return S_samples, logit_list


    def sample_hidden_posterior(self, encoded_sents, S_samples):
        """
        Sample the hidden state Z posterior given the encoded data
        Args
            encoded_sents (Tensor, [num_sents, batch, encoded dim])  
            S_samples (Tensor, [num_sents, batch, num_classes]) : the switching state posterior samples
        Returns
            hidden samples (Tensor, [num_sents, batch, hidden dim]) : samples from postieror dyncamics distribtion
            mean (Tensor, [num_sents, batch, hidden dim]) : mean of posterior (for KL calc)
        """
        batch_size = input.size(1)
        num_sents = input.size(0)
        Z_samples = []
        means = []
        prior_means = []
        logvars = []

        prev_z = Variable(torch.zeros(batch_size, self.hidden_size))
        for i in range(num_sents):
            context_vect = utils.get_context_vector(encoded_sents, i, future=True)
            target_vect = encoded_sents[i, :, :]
            switch_state = S_samples[i]
            mean, logvar, prior_mean = dynamics_posterior_factor(prev_z, switch_state, torch.cat([target_vect, context_vect], dim=1))
            z_sample = utils.normal_sample(mean, logvar)
            Z_samples.append(z_sample)
            means.append(mean)
            prior_means.append(prior_mean)
            logvars.append(logvar)
            prev_z = z_sample

        Z_samples = torch.stack(Z_samples, dim=0) #[num_sents, batch, hidden]
        means= torch.stack(means, dim=0) #[num_sents, batch, hidden]
        prior_means= torch.stack(prior_means, dim=0) #[num_sents, batch, hidden]
        logvars= torch.stack(logvars, dim=0) #[num_sents, batch, hidden]
        return Z_samples, means, prior_means, logvars




