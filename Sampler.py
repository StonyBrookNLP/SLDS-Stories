##############################################
#    Gibbs Sampler for Filling in the Blanks
#    
#   
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
from masked_cross_entropy import masked_cross_entropy
from data_utils import EOS_TOK, SOS_TOK, PAD_TOK, transform


class GibbsSampler():
    def __init__(self, model, vocab, max_iterations=3000, use_bias=False, use_cuda=False):
        """
        Args:
            model (SLDS)
        """
        self.model = model
        self.use_cuda= use_cuda
        self.max_iterations = max_iterations
        self.vocab = vocab
        self.use_bias = use_bias #Whether or not bias is included in the transition dynamics formula
        if self.use_bias:
            print("Using a bias term")



    def aggregate_gibbs_sample(self, input, switch_states, missing_sents, lengths, burn_in=500, num_samples=1000, trans_matrix=None):
        print("Gibbs Sampler, Sample all Z, Then Outputs")
    
        batch_size = 1
        num_sents = input.size(0)

        if switch_states is not None:
            one_hots = torch.zeros(num_sents, 1, self.model.num_states).cuda() if self.use_cuda else torch.zeros(num_sents,1,self.model.num_states)
            switch_states_oneh = one_hots.scatter(2, switch_states, 1) #convert to one_hot vectors, of shape [num_sents, 1, num_classes]
        else: #Calculate the discrete state for non missing sentences
            print("FILLIN")
            encoded_sents = self.encode_sents(input, lengths)
            _, state_logits= self.model.sample_switch_posterior(torch.stack(encoded_sents,dim=0), gumbel_temp=0.1)
            all_states= torch.argmax(state_logits,dim=2).unsqueeze(dim=2) #[sents, 1,1]

            #ehidden = self.model.sentence_encode_rnn.initHidden(1)
           # encoded_sent_first, _ = self.encode_single(input[0], lengths[0], ehidden) 
           # encoded_sent_last, _ = self.encode_single(input[-1], lengths[-1], ehidden) 

          #  _, state_logits= self.model.sample_switch_posterior(torch.stack([encoded_sent_first,encoded_sent_last],dim=0), gumbel_temp=0.1)
          #  fl = torch.argmax(state_logits,dim=2)
          #  first_state = fl[0].squeeze().item()
          #  last_state = fl[1].squeeze().item()
          #  all_states = [[[first_state]]]
          #  for i in range(3):
          #      prev_state = all_states[i][0][0]
          #      next_state = np.argmax(trans_matrix[prev_state])
          #      all_states.append([[next_state]])
          #  all_states.append([[last_state]])
          #  all_states = torch.LongTensor(all_states)
            one_hots = torch.zeros(num_sents, 1, self.model.num_states).cuda() if self.use_cuda else torch.zeros(num_sents,1,self.model.num_states)
            switch_states_oneh = one_hots.scatter(2,all_states, 1) #convert to one_hot vectors, of shape [num_sents, 1, num_classes]


                

        #Initialize missing values
        zs = self.init_zs_2(switch_states_oneh, input, missing_sents, lengths)
        outputs, encoded_sents, lengths = self.init_text(input, missing_sents, zs, lengths) #outputs is list of list of indices for each sentence

        print("Initial Sentences")
        for i, sent in enumerate(outputs):
            print("{}".format(transform(outputs[i], self.vocab.itos)))
        print("--------------------\n\n")

        min_cross_entropy = 1000.0
        best_outputs = outputs

        for i in range(burn_in + num_samples):
            zs = self.sample_zs(zs, outputs, switch_states_oneh, encoded_sents)
            outputs, newlengths, last_logits = self.sample_outputs(outputs, zs, missing_sents)
            encoded_sents = self.encode_sents([torch.LongTensor(x).unsqueeze(dim=0) for x in outputs], newlengths) #Re-encode the newoutputs

            if (i+1) > burn_in:
                cross_entropy = masked_cross_entropy(last_logits, torch.LongTensor(outputs[-1][1:]).unsqueeze(dim=0), newlengths[-1] - 1, use_cuda=self.use_cuda).item()
                #cross_entropy = masked_cross_entropy(last_logits, torch.LongTensor(outputs[3][1:]).unsqueeze(dim=0), newlengths[3] - 1, use_cuda=self.use_cuda).item()
                if cross_entropy < min_cross_entropy:
                    min_cross_entropy = cross_entropy
                    best_outputs = outputs
                    print("Cross Entropy: {}".format(min_cross_entropy))
                    for j, sent in enumerate(best_outputs):
                        print("{}".format(transform(best_outputs[j], self.vocab.itos)))

                    print("--------------------\n\n")
                   
                

#            for j, sent in enumerate(outputs):
#                print("{}".format(transform(outputs[j], self.vocab.itos)))

        print("--------------------\n\n")

        return best_outputs

    def gibbs_sample(self, input, switch_states, missing_sents, lengths): #This is good
        """
        Args
            missing_sents ([int] an integer list indicating which sentences are missing
            input (Tensor, [num_sents, batch, seq_len]) : Tensor of input ids for the embeddings lookup
            switch_states (Tensor, [num_sents, batch, 1] : Tensor giving the class id of the values of the switch states)
            lengths (Tensor, [num_sents, batch]

            #INIT States first
            #INit Z's
            #Init Text

            #ALGO
            1. Put in the sentences and get the hidden state encodings for each individual sentence?
            2. iterate through z's
            3. iterate through X's
        """

        #batch, seq_lens = story_batches.combine_story(story) #should return batch tensor [num_sents, batch, seq_len] and seq_lens [num_sents, batch]
        #state_targets= story_batches.combine_sentiment_labels(story, use_cuda=use_cuda) #state_targets is [numsents, batch, 1]

        print("Gibbs Sampler, Sample all Z, Then Outputs")
    
        batch_size = 1
        num_sents = input.size(0)

        one_hots = torch.zeros(num_sents, 1, self.model.num_states).cuda() if self.use_cuda else torch.zeros(num_sents,1,self.model.num_states)
        switch_states_oneh = one_hots.scatter(2, switch_states, 1) #convert to one_hot vectors, of shape [num_sents, 1, num_classes]

        #Initialize missing values
#        zs = self.init_zs(switch_states_oneh) #zs is a python list of Tensors of shape (1, hidden_size)

        zs = self.init_zs_2(switch_states_oneh, input, missing_sents, lengths) #This is good
#        zs = [torch.randn([1,self.model.hidden_size]) for _ in range(num_sents)]
        outputs, encoded_sents, lengths = self.init_text(input, missing_sents, zs, lengths) #outputs is list of list of indices for each sentence

        for i, sent in enumerate(outputs):
            print("{}".format(transform(outputs[i], self.vocab.itos)))
        print("--------------------\n\n")


        for i in range(self.max_iterations):
            zs = self.sample_zs(zs, outputs, switch_states_oneh, encoded_sents)
            outputs, newlengths,_ = self.sample_outputs(outputs, zs, missing_sents)
            encoded_sents = self.encode_sents([torch.LongTensor(x).unsqueeze(dim=0) for x in outputs], newlengths) #Re-encode the newoutputs

            if i % 10 == 0:
                for j, sent in enumerate(outputs):
                    print(i)
                    print("{}".format(transform(outputs[j], self.vocab.itos)))
                print("--------------------\n\n")


        return outputs

    def sample_outputs(self, outputs, zs, missing_sents):
        dhidden=None
        ret_outputs = []
        num_sents = len(zs)
        logits = None

        for i in range(num_sents):

            if i not in missing_sents:
                input_tensor = torch.LongTensor(outputs[i]).unsqueeze(dim=0)
                logits, dhidden = self.model.data_liklihood_factor(input_tensor, zs[i], dhidden, torch.LongTensor([len(outputs[i])])) 
                ret_outputs.append(outputs[i])
            else:
                sent_out, dhidden = self.model.greedy_decode(zs[i], dhidden=dhidden)
                sent_out.append(self.model.eos_idx) #add the eos token 
                sent_out = [self.model.sos_idx] + sent_out
                ret_outputs.append(sent_out)

        lengths = [len(x) for x in ret_outputs]
        lengths = torch.LongTensor(lengths).unsqueeze(dim=1)
        return ret_outputs, lengths, logits


    def sample_zs(self, zs, outputs, switch_states_oneh, encoded_sents):
        num_sents = len(zs)
        all_mean_weights = torch.stack([x.weight for x in self.model.dynamics_mean]) #[num_states X hidden X hidden]
        if self.use_bias:
            all_mean_bias = torch.stack([x.bias for x in self.model.dynamics_mean]) #[num_states X hidden]


        for i in range(num_sents):

            if i == 0:
                prev_z = torch.zeros([1,self.model.hidden_size]).cuda() if self.use_cuda else torch.zeros([1,self.model.hidden_size])
            else:
                prev_z = zs[i-1]

            #Get the values from the posterior
            mean_posterior, logvar_posterior, _, _ = self.model.dynamics_posterior_factor(prev_z, switch_states_oneh[i], encoded_sents[i])
            inv_var_posterior = torch.exp(-1*logvar_posterior)
                    
            #Get the future information (if not at the last in seq)
            if i < num_sents - 1: #If not the last in seq
                next_state = switch_states_oneh[i+1].squeeze(dim=0) #[num_classes]
                next_z = zs[i+1]
                A = torch.einsum('ijk,i->jk', [all_mean_weights, next_state])  #[hidden X hidden]
                logvar_s = torch.einsum('ij,i->j', [self.model.dynamics_logvar, next_state]) #[hidden]
                inv_var_s = torch.exp(-1*logvar_s)

                inv_var_s_A = inv_var_s.unsqueeze(dim=1)*A
                inv_var = torch.matmul(A.t(), inv_var_s_A) + torch.diag(inv_var_posterior.squeeze())
                var = torch.inverse(inv_var)

                if self.use_bias:
                    bias_param = torch.einsum('ij,i->j', [all_mean_bias, next_state]).unsqueeze(dim=0)
                    term2 = torch.matmul(next_z - bias_param, inv_var_s_A) + (mean_posterior * inv_var_posterior)
                else:
                    term2 = torch.matmul(next_z, inv_var_s_A) + (mean_posterior * inv_var_posterior)

                mean = torch.matmul(term2, var).squeeze()
            else:
                mean = mean_posterior.squeeze()
                var = torch.diag(torch.exp(logvar_posterior).squeeze())
            
            sampling_dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=var)
            zs[i] = sampling_dist.sample().unsqueeze(dim=0)
        return zs


    def init_zs(self, switch_states_oneh):
        zs = []
        prev_z = torch.zeros([1,self.model.hidden_size]).cuda() if self.use_cuda else torch.zeros([1,self.model.hidden_size])
        num_sents = switch_states_oneh.size(0)

        for i in range(num_sents):
            prior_mean, prior_logvar = self.model.dynamics_factor(prev_z, switch_states_oneh[i])
            z_i = utils.normal_sample(prior_mean, prior_logvar, use_cuda=self.use_cuda)
            zs.append(z_i)
            prev_z = z_i 

        return zs

    def init_zs_2(self, switch_states_oneh, input, missing_sents, seq_lens):
        zs = []
        prev_z = torch.zeros([1,self.model.hidden_size]).cuda() if self.use_cuda else torch.zeros([1,self.model.hidden_size])
        num_sents = switch_states_oneh.size(0)

        ehidden = self.model.sentence_encode_rnn.initHidden(1)
        for i in range(num_sents):
            if i not in missing_sents:
                print("Posterior")
                encoded_sent_i, ehidden = self.encode_single(input[i], seq_lens[i], ehidden) 
                prior_mean, prior_logvar, _, _ = self.model.dynamics_posterior_factor(prev_z, switch_states_oneh[i], encoded_sent_i)
            else:
                prior_mean, prior_logvar = self.model.dynamics_factor(prev_z, switch_states_oneh[i])

            z_i = utils.normal_sample(prior_mean, prior_logvar, use_cuda=self.use_cuda)
            zs.append(z_i)
            prev_z = z_i 

        return zs


    def init_text(self, input, missing_sents, zs, lengths):
        """
        Init both encoded sentences and the actual missing text
        """
        encoded_sents = []
        outputs = [] #Init the outputs so that the missing values have nothing and non missing values have something 
        num_sents = input.size(0)

        ehidden = self.model.sentence_encode_rnn.initHidden(1)
        dhidden = None
        for i in range(num_sents):
            if i not in missing_sents:
                encoded_sent_i, ehidden = self.encode_single(input[i], lengths[i], ehidden) 
                _, dhidden = self.model.data_liklihood_factor(input[i], zs[i], dhidden, lengths[i])
                sent_out = input[i].squeeze().tolist()
                sent_out = [x for x in sent_out if x != self.model.pad_idx] #remove pads
                encoded_sents.append(encoded_sent_i)
                outputs.append(sent_out)
            else:
                sent_out, dhidden = self.model.greedy_decode(zs[i], dhidden=dhidden)
                sent_out.append(self.model.eos_idx) #add the eos token 
                sent_out = [self.model.sos_idx] + sent_out
                lengths[i,0] = len(sent_out) #update the lengths
                outputs_tensor = torch.LongTensor(sent_out).unsqueeze(dim=0)
                encoded_sent_i, ehidden = self.encode_single(outputs_tensor, lengths[i], ehidden)
                encoded_sents.append(encoded_sent_i)
                outputs.append(sent_out)

        return outputs, encoded_sents, lengths
    
    def encode_single(self, input, seq_lens, ehidden):
        """
        Encode each sentence into vector representation
        Args
            input (Tensor, [batch, seq_len]) : Tensor of input ids for the embeddings lookup
            seq_lens (Tensor [batch]) : Store the sequence lengths for each batch for packing
        Returns
            encoded sentences (Tensor, [num_sents, batch, vocab size]) : logits for the output words
        """

        enc_output, _ = self.model.sentence_encode_rnn(input, ehidden, seq_lens, use_packed=False)
        last_output = gather_last(enc_output, seq_lens, use_cuda=self.use_cuda) #[batch, encoder_dim]

        ehidden = last_output.unsqueeze(dim=0) #new so pads are not processed in encoding

        return last_output, ehidden

    def encode_sents(self, input, seq_lens):
        """
        Encode each sentence into vector representation
        Args
            input (List(Tensor([batch X seq]))) : Tensor of input ids for the embeddings lookup
            seq_lens (Tensor [num_sents, batch]) : Store the sequence lengths for each batch for packing
        Returns
            encoded sentences (Tensor, [num_sents, batch, vocab size]) : logits for the output words
        """

        num_sents = len(input)

        encoded_sents = []
        ehidden = self.model.sentence_encode_rnn.initHidden(1)

        for i in range(num_sents):
            enc_output, _ = self.model.sentence_encode_rnn(input[i], ehidden, seq_lens[i], use_packed=False)
            last_output = gather_last(enc_output, seq_lens[i], use_cuda=self.use_cuda) #[batch, encoder_dim]

            ehidden = last_output.unsqueeze(dim=0) #new so pads are not processed in encoding
            encoded_sents.append(last_output)

        return encoded_sents




    def average_aggregate_gibbs_sample(self, input, switch_states, missing_sents, lengths, burn_in=1000, num_samples=2000):
        print("Gibbs Sampler, Sample all Z, Then Outputs")
    
        batch_size = 1
        num_sents = input.size(0)

        one_hots = torch.zeros(num_sents, 1, self.model.num_states).cuda() if self.use_cuda else torch.zeros(num_sents,1,self.model.num_states)
        switch_states_oneh = one_hots.scatter(2, switch_states, 1) #convert to one_hot vectors, of shape [num_sents, 1, num_classes]

        #Initialize missing values
#        zs = self.init_zs(switch_states_oneh) #zs is a python list of Tensors of shape (1, hidden_size)

        zs = self.init_zs_2(switch_states_oneh, input, missing_sents, lengths)
        average_zs = [torch.zeros([1,self.model.hidden_size]) for _ in range(num_sents)]
#        zs = [torch.randn([1,self.model.hidden_size]) for _ in range(num_sents)]
        outputs, encoded_sents, lengths = self.init_text(input, missing_sents, zs, lengths) #outputs is list of list of indices for each sentence

        print("Initial Sentences")
        for i, sent in enumerate(outputs):
            print("{}".format(transform(outputs[i], self.vocab.itos)))
        print("--------------------\n\n")


        samples_added = 0
        for i in range(burn_in + num_samples):
            zs = self.sample_zs(zs, outputs, switch_states_oneh, encoded_sents)
            outputs, newlengths = self.sample_outputs(outputs, zs, missing_sents)
            encoded_sents = self.encode_sents([torch.LongTensor(x).unsqueeze(dim=0) for x in outputs], newlengths) #Re-encode the newoutputs

            if (i+1) > burn_in and (i+1) % 5 == 0:
                average_zs = [average_zs[idx] + (zs[idx] / (num_samples / 5.0)) for idx in range(num_sents)]
                samples_added +=1
               # print("Iteration {}, Past Burn In Phase".format(i))

        print(samples_added)
        print(num_samples / 5.0)
                
        print("Getting output of Average Z")
        last_outputs, newlengths = self.sample_outputs(outputs, average_zs, missing_sents)

        for j, sent in enumerate(last_outputs):
            print("{}".format(transform(last_outputs[j], self.vocab.itos)))

        print("--------------------\n\n")

        return last_outputs

