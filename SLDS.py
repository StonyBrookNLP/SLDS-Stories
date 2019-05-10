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
from EncDec import Encoder, Decoder, gather_last, sequence_mask
from data_utils import EOS_TOK, SOS_TOK, PAD_TOK


class SLDS(nn.Module):
    def __init__(self, hidden_size, rnn_hidden_size, embd_size, vocab, trans_matrix, layers=1, pretrained=False, dropout=0.0, use_bias=True, use_cuda=False):
            """
            Args:
                hidden_size (int) : size of hidden vector z 
                embd_size (int) : size of word embeddings
                vocab (torchtext.Vocab) : vocabulary object
                trans_matrix (Tensor, [num states, num states]) : Transition matrix probs for switching markov chain
                pretrained (bool) : use pretrained word embeddings?

            """
            super(SLDS, self).__init__()
            self.hidden_size = hidden_size
            self.encoded_data_size = self.dec_hsize = rnn_hidden_size #Vector size to use whenever encoding text into a %&#ing vector

            self.trans_matrix = trans_matrix
            self.num_states = trans_matrix.shape[0]
            self.embd_size=embd_size
            self.layers = layers #Right now, encoder needs to have just 1 layer
            self.use_cuda = use_cuda 
            self.bias = use_bias
            print("Using Bias: {}".format(self.bias))
            
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


            self.dynamics_mean = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias) for _ in range(self.num_states)])

            if use_cuda:
                self.dynamics_logvar = torch.nn.Parameter(torch.zeros([self.num_states, self.hidden_size]).cuda()) #diagnol covariance [states, hidden] 
            else:
                self.dynamics_logvar = torch.nn.Parameter(torch.zeros([self.num_states, self.hidden_size])) #diagnol covariance [states, hidden] 

            print("Init Logvar Kaiming Uniform SLDS")
            nn.init.kaiming_uniform_(self.dynamics_logvar, a=math.sqrt(5.0)) #This is the default for linear layers too


            self.sentence_encode_rnn= Encoder(self.embd_size, self.encoded_data_size, in_embedding, "GRU", self.layers, bidir=False, use_cuda=use_cuda)
            self.z2dec = nn.Linear(self.hidden_size, self.dec_hsize*self.layers, bias=False) #Convert z to initial hidden state for decoder
            self.liklihood_rnn= Decoder(self.embd_size + self.hidden_size, self.dec_hsize, out_embedding, "GRU", self.layers, use_cuda=use_cuda, dropout=dropout)
            self.liklihood_logits= nn.Linear(self.dec_hsize, self.vocab_size, bias=False) #Weights to calculate logits, out [batch, vocab_size]

            #Variational Parameters
            self.dynamics_posterior_network = nn.Linear(self.encoded_data_size, self.hidden_size, bias=False)
            self.dynamics_variance_posterior_network = nn.Linear(self.encoded_data_size, self.hidden_size, bias=False)
            self.state_posterior_network = nn.Sequential(
                                                nn.Linear(self.encoded_data_size*2, self.hidden_size),
                                                nn.Tanh(),
                                                nn.Linear(self.hidden_size, self.num_states)
                                              )

            if use_cuda:
                self.dynamics_mean = self.dynamics_mean.cuda()
                self.trans_matrix = self.trans_matrix.cuda()
                self.sentence_encode_rnn = self.sentence_encode_rnn.cuda()
                self.z2dec = self.z2dec.cuda()
                self.liklihood_rnn = self.liklihood_rnn.cuda()
                self.liklihood_logits = self.liklihood_logits.cuda()
                self.dynamics_posterior_network = self.dynamics_posterior_network.cuda()
                self.dynamics_variance_posterior_network = self.dynamics_variance_posterior_network.cuda()
                self.state_posterior_network = self.state_posterior_network.cuda()

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
        batch_size = input.size(1)
        num_sents = input.size(0)

        encoded_sents = self.encode_sentences(input, seq_lens) #[num_sents, batch, encoder dim)
        S_samples, state_logits = self.sample_switch_posterior(encoded_sents, gumbel_temp=gumbel_temp)

        if state_labels is not None:
            one_hots = torch.zeros(S_samples.shape).cuda() if self.use_cuda else torch.zeros(S_samples.shape)
            S_samples = one_hots.scatter(2, state_labels, 1) #convert to one_hot vectors
                

        Z_samples, Z_means, prior_means, logvars, prior_vars = self.sample_hidden_posterior(encoded_sents, S_samples) #[num_sents, batch, hidden_dim]

       
        #Now Evaluate the Liklihood for each sentence

        dhidden=None
        data_logits = []
        for i in range(num_sents):
            logits, dhidden = self.data_liklihood_factor(input[i,:,:], Z_samples[i], dhidden, seq_lens[i]) # USE IF PASSING PREV HIDDEN STATE TO NEXT
          #  logits, dhidden = self.data_liklihood_factor(input[i,:,:], Z_samples[i]) #logits is [batch, seq, num_classes]
            data_logits.append(logits)

        data_logits = torch.stack(data_logits, dim=0) #[num_sents, batch, seq, num classes]

        #Calc KL terms
        Z_kl = self.z_kl_divergence(Z_means, prior_means, logvars, prior_vars)
        state_kl = self.state_kl_divergence(F.softmax(state_logits, dim=2), self.state_factor(S_samples))

        print(torch.mean((Z_means[1] - prior_means[1])*(Z_means[1] - prior_means[1])))

        return data_logits, state_logits, Z_kl, state_kl

    def z_kl_divergence(self, Z_means, prior_means, logvars, prior_logvars):
        """
        Calcuate the kl_devergence for the transition (z) distributions
        Calculate kl as: 
        (prior_mean- z_mean)^T exp(-logvars) (prior_means-z_mean)
        Args
            z_means, prior_means, logvars (Tensor [num_sents, batch, hidden_dim])

        Return:
            kl (Tensor [batch_size])
        """
        num_sents = Z_means.size(0)
        batch_size = Z_means.size(1)

        inverse_prior_var = torch.exp(-1*prior_logvars)
        variance = torch.exp(logvars)

        if self.use_cuda:
            kl = Variable(torch.zeros(batch_size).cuda())
        else:
            kl = Variable(torch.zeros(batch_size))

        for i in range(num_sents): #Calculate for each sentence
            term1 = torch.sum(variance[i]*inverse_prior_var[i], dim=1) 
            term2 = torch.sum(torch.log(torch.exp(prior_logvars[i])), dim=1) - torch.sum(torch.log(variance[i]), dim=1)
            
            diff = prior_means[i] - Z_means[i]
            diff_squared = diff*diff
            kl += (1.0/2.0)*(torch.diag(torch.matmul(inverse_prior_var[i], diff_squared.transpose(0,1))) + term1 + term2 - self.hidden_size)

        return kl


    def state_kl_divergence(self, q_probs, p_probs):
        """
        Calcuate the kl_devergence for the switching state distributions
        (prior_mean- z_mean)^T exp(-logvars) (prior_means-z_mean)
        Args
            q_probs, (Tensor [num_sents, batch, num_classes])
            p_probs, (Tensor [num_sents, batch, num_classes])
        
        NEED TO FINISH THIS ONE
        Return:
            kl (Tensor [batch_size])
        """
        
        num_sents = q_probs.size(0)
        batch_size = q_probs.size(1)

        if self.use_cuda:
            kl = Variable(torch.zeros(batch_size).cuda())
        else:
            kl = Variable(torch.zeros(batch_size))

        for i in range(num_sents): #Calculate for each sentence
            kl_i = q_probs[i]*(torch.log(q_probs[i]) - torch.log(p_probs[i]))
            kl += kl_i.sum(dim=1)
#
        return kl

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
        if self.bias:
            all_mean_bias = torch.stack([x.bias for x in self.dynamics_mean]) #[num_states X hidden]
            bias_param = torch.einsum('ij,bi->bj', [all_mean_bias, switch_state])  #[batch X hidden]

        #crate single matrix as a convex comb of all dynamics matrix, using switch state as weights, 1 matrix for each batch
        mean_param = torch.einsum('ijk,bi->bjk', [all_mean_weights, switch_state])  #[batch X hidden X hidden]
        logvar_param = torch.einsum('ij,bi->bj', [self.dynamics_logvar, switch_state]) #[batch X hidden]

        if self.bias:
            means = torch.einsum('bij,bj->bi',[mean_param, prev_z]) + bias_param #batch X hidden]
        else:
            means = torch.einsum('bij,bj->bi',[mean_param, prev_z]) #batch X hidden]
        return (means, logvar_param)

    
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
        means_prior, logvar_prior = self.dynamics_factor(prev_z, switch_state)

        network_input = encoded_data
        mean_residual = self.dynamics_posterior_network(network_input)
        logvar_residual = self.dynamics_variance_posterior_network(network_input)

        ###############################################
        inverse_prior_var = torch.exp(-1*logvar_prior)
        inverse_var= torch.exp(-1*logvar_residual)

        product_variance = 1.0 / (inverse_prior_var + inverse_var)
        product_mean = (inverse_var*mean_residual + inverse_prior_var*means_prior)*product_variance

        means = product_mean
        logvar = torch.log(product_variance)
        ###############################################

        return (means, logvar, means_prior, logvar_prior)

    def state_factor(self, curr_switch_state):
        """
        Return the probabilities for the transition distribution given the current switch state
        Args:
            curr_switch_state (Tensor [num_sents, batch, num_states]) : can be a one hot or a probabilistic vector, if the latter, transition dist is a mixture
        """
        num_sents = curr_switch_state.size(0)
#        state_transitions = [torch.matmul(curr_switch_state[i], self.trans_matrix) for i in range(num_sents)]
#        state_transitions = [torch.zeros(curr_switch_state.shape[1], curr_switch_state.shape[2]).cuda() + torch.Tensor([0.30, 0.02, 0.02, 0.30, 0.02, 0.02, 0.02, 0.30 ]).cuda()] + [torch.matmul(curr_switch_state[i+1].detach(), self.trans_matrix) for i in range(num_sents-1)]
        state_transitions = [torch.zeros(curr_switch_state.shape[1], curr_switch_state.shape[2]).cuda() + torch.Tensor([0.30, 0.30, 0.40]).cuda()] + [torch.matmul(curr_switch_state[i+1].detach(), self.trans_matrix) for i in range(num_sents-1)]
        state_transitions = torch.stack(state_transitions, dim=0) #[num_sents, batch, encoder dim)

        return state_transitions

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
    def data_liklihood_factor(self, data, curr_z, dhidden=None, lengths=None):
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
            dhidden = F.tanh(self.z2dec(curr_z)) #output shape [batch, layers*hidden]
            dhidden = dhidden.view(-1, self.layers, self.dec_hsize).transpose(0,1).contiguous() #[layers, batch, hiddensize]

        logits = []
        for i in range(data.size(1)-1): #dont process last (the eos)

            dec_input = data[:, i]

            #dec_output is [batch, hidden_dim]
            dec_output, dhidden = self.liklihood_rnn(dec_input, dhidden, concat=curr_z) 
            logits_t = self.liklihood_logits(dec_output)
            logits += [logits_t]

            dhidden_list += [dhidden.transpose(0,1)] #list stores [batch, layers, hiddensize]
 
        logits = torch.stack(logits, dim=1) #DIMENSION BE [batch x seq x num_classes]

        dhidden_list = torch.stack(dhidden_list, dim=1) #[batch x seq x layers x hidden_size]
        dhidden_list = dhidden_list.view(dhidden_list.shape[0], dhidden_list.shape[1], -1) #[batch, seq, layers*hidden_size]
        last_dhidden = gather_last(dhidden_list, lengths - 1, use_cuda=self.use_cuda)
        last_dhidden = last_dhidden.view(-1, self.layers, self.dec_hsize).transpose(0,1).contiguous()  #[layers, batch, hiddensize]

        return logits, last_dhidden

    def encode_sentences(self, input, seq_lens):
        """
        Encode each sentence into vector representation
        Args
            input (Tensor, [num_sents, batch, seq_len]) : Tensor of input ids for the embeddings lookup
            seq_lens (Tensor [num_sents, batch]) : Store the sequence lengths for each batch for packing
        Returns
            encoded sentences (Tensor, [num_sents, batch, vocab size]) : logits for the output words
        """

        num_sents = input.size(0)
        batch_size = input.size(1)

        encoded_sents = []
        ehidden = self.sentence_encode_rnn.initHidden(batch_size)

        for i in range(num_sents):

            enc_output, _ = self.sentence_encode_rnn(input[i,:,:], ehidden, seq_lens[i], use_packed=False)
            last_output = gather_last(enc_output, seq_lens[i], use_cuda=self.use_cuda) #[batch, encoder_dim]

            ehidden = last_output.unsqueeze(dim=0) #so pads are not processed in encoding, for this reason cant have more than 1 layer encoder
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
            context_vect = utils.get_context_vector(encoded_sents, i, future=False, use_cuda=self.use_cuda)
            target_vect = encoded_sents[i, :, :]
            logits = self.state_posterior_factor(torch.cat([target_vect, context_vect], dim=1)) #[batch, num classes]
            sample = utils.gumbel_softmax_sample(logits, temp=gumbel_temp, use_cuda=self.use_cuda)
            S_samples.append(sample)
            logit_list.append(logits)

        S_samples = torch.stack(S_samples, dim=0) #[num_sents, batch, num classes]
        logit_list = torch.stack(logit_list, dim=0)
        return S_samples, logit_list


    def sample_hidden_posterior(self, encoded_sents, S_samples, sample_from_prior=False, eps=None):
        """
        Sample the hidden state Z posterior given the encoded data
        Args
            encoded_sents (Tensor, [num_sents, batch, encoded dim])  
            S_samples (Tensor, [num_sents, batch, num_classes]) : the switching state posterior samples
            sample from prior : whether or not to sample from prior (rather than posterior), use for generation
        Returns
            hidden samples (Tensor, [num_sents, batch, hidden dim]) : samples from postieror dyncamics distribtion
            mean (Tensor, [num_sents, batch, hidden dim]) : mean of posterior (for KL calc)
            prior mean (Tensor, [num_sents, batch, hidden dim]) : mean of posterior under prior (for KL calc)
            logvars (Tensor, [num_sents, batch, hidden dim]) : logvars (for KL calc)
            
        """
        batch_size = encoded_sents.size(1)
        num_sents = encoded_sents.size(0)
        Z_samples = []
        means = []
        prior_means = []
        prior_vars = []
        logvars = []

        if self.use_cuda:
            prev_z = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
        else:
            prev_z = Variable(torch.zeros(batch_size, self.hidden_size))

        for i in range(num_sents):
            context_vect = utils.get_context_vector(encoded_sents, i, future=True, use_cuda=self.use_cuda)
            context_vect_past = utils.get_context_vector(encoded_sents, i, future=False, use_cuda=self.use_cuda)

            target_vect = encoded_sents[i, :, :]
            switch_state = S_samples[i]
            mean, logvar, prior_mean, prior_var = self.dynamics_posterior_factor(prev_z, switch_state, target_vect) 

            if sample_from_prior and i >= 0:
                print("sample from prior")
                
#                z_sample = utils.normal_sample(prior_mean, prior_var, use_cuda=self.use_cuda) 
                if i == 0:
                    z_sample = utils.normal_sample_deterministic(mean, logvar, eps[i], use_cuda=self.use_cuda) 
                else:
                    z_sample = utils.normal_sample_deterministic(prior_mean, prior_var, eps[i], use_cuda=self.use_cuda) 
            else:
                z_sample = utils.normal_sample(mean, logvar, use_cuda=self.use_cuda)
                #print(z_sample)
            Z_samples.append(z_sample)
            means.append(mean)
            prior_means.append(prior_mean)
            logvars.append(logvar)
            prior_vars.append(prior_var)
            prev_z = z_sample

        Z_samples = torch.stack(Z_samples, dim=0) #[num_sents, batch, hidden]
        means= torch.stack(means, dim=0) #[num_sents, batch, hidden]
        prior_means= torch.stack(prior_means, dim=0) #[num_sents, batch, hidden]
        prior_vars= torch.stack(prior_vars, dim=0) #[num_sents, batch, hidden]
        logvars= torch.stack(logvars, dim=0) #[num_sents, batch, hidden]
        return Z_samples, means, prior_means, logvars, prior_vars


    def generative_params(self):
        names = ['dynamics_mean']
        p_list = [x[1].parameters() for x in self.named_children() if x[0] in names] + [iter([self.dynamics_logvar])]
        return itertools.chain(*p_list)

    def variational_params(self):
#        names = ['sentence_encode_rnn', 'dynamics_posterior_network', 'dynamics_variance_posterior_network', 'state_posterior_network']
        names = ['sentence_encode_rnn', 'dynamics_posterior_network', 'dynamics_variance_posterior_network', 'state_posterior_network' , 'z2dec', 'liklihood_rnn', 'liklihood_logits']
        p_list = [x[1].parameters() for x in self.named_children() if x[0] in names]
        return itertools.chain(*p_list)

    def greedy_decode(self, curr_z, max_decode=30, dhidden=None):
        """
        Output the logits at each timestep (the data liklihood outputs from the rnn)
        Args:
            data (Tensor, [batch, seq_len]) vocab ids of the data
            curr_z (Tensor, [batch, hidden_size]) 
        Ret:
            outputs - list of indicies
        """

        if dhidden is None:
            dhidden = F.tanh(self.z2dec(curr_z)) #output shape [batch, layers*hidden]
            dhidden = dhidden.view(-1, self.layers, self.dec_hsize).transpose(0,1).contiguous() #[layers, batch, hiddensize]
            
        outputs = []
        prev_output = Variable(torch.LongTensor(1).zero_() + self.sos_idx)

        for i in range(max_decode): 
            dec_input = prev_output

            dec_output, dhidden = self.liklihood_rnn(dec_input, dhidden, concat=curr_z) 
            logits_t = self.liklihood_logits(dec_output)

            #dec_output is [batch, hidden_dim]

            probs = F.log_softmax(logits_t, dim=1)
            top_vals, top_inds = probs.topk(1, dim=1)

          #  logits_t=utils.top_k_logits(logits_t,k=10)
          #  probs = F.softmax(logits_t, dim=1)
          #  top_inds = torch.multinomial(probs, 1)

            if top_inds.squeeze().item() == self.eos_idx:
                break

            outputs.append(top_inds.squeeze().item())
            prev_output = top_inds[0]


        return outputs, dhidden

    def reconstruct(self, input, seq_lens, eps=None, initial_sents=0):
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

        encoded_sents = self.encode_sentences(input, seq_lens) #[num_sents, batch, encoder dim)
        S_samples, state_logits = self.sample_switch_posterior(encoded_sents, gumbel_temp=0.1)
        Z_samples, Z_means, prior_means, logvars, prior_logvars = self.sample_hidden_posterior(encoded_sents, S_samples, sample_from_prior=True, eps=eps) #[num_sents, batch, hidden_dim]
        for i in range(num_sents):
            print(torch.mean((Z_means[i] - prior_means[i])*(Z_means[i] - prior_means[i])))
            print(torch.mean(Z_means[i] - prior_means[i]))
            print(torch.mean(torch.exp(logvars[i]) - torch.exp(prior_logvars[i])))
            print("--")
        
        print(F.softmax(state_logits, dim=2))
        print(torch.exp(logvars).mean())
       
        #Now Evaluate the Liklihood for each sentence


        dhidden=None
        outputs = []
        for i in range(num_sents):

            if i < initial_sents:
                print("Initial Sent {}".format(i))
                _, dhidden = self.data_liklihood_factor(input[i,:,:], Z_samples[i], dhidden, seq_lens[i]) # USE IF PASSING PREV HIDDEN STATE TO NEXT
                sent_out = input[i, :, :].squeeze().tolist()
            else:
                sent_out, dhidden= self.greedy_decode(Z_samples[i], dhidden=dhidden) #USE IF PASSING PREV HIDDEN STATE TO NEXT
               # sent_out, dhidden= self.greedy_decode(Z_samples[i])

            outputs.append(sent_out)

        return outputs

    def set_use_cuda(self, value):
        self.use_cuda = value
        self.liklihood_rnn.use_cuda = value
        self.sentence_encode_rnn.use_cuda = value



