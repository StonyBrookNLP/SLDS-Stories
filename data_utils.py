################################
# Objects for storing and iterating over dataset objects
# Uses texttorch stuff, so make sure thats installed 
################################
import torch 
import torch.nn as nn
import numpy as np
import math
import csv
import spacy
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torchtext.data as ttdata
import torchtext.datasets as ttdatasets
from torchtext.vocab import Vocab
from collections import defaultdict, Counter

#Reserved Special Tokens
PAD_TOK = "<pad>"
SOS_TOK = "<sos>" #start of sentence
EOS_TOK = "<eos>" #end of sentence 
UNK_TOK = "<unk>"

POS_LABEL = "POS"
NEG_LABEL = "NEG"
NEU_LABEL = "NEU"

#These are the values that should be used during evalution to keep things consistent
MIN_EVAL_SEQ_LEN = 8
MAX_EVAL_SEQ_LEN = 50 

#A Field for a single sentence from the Book Corpus (or any other corpus with a single item per line)

#PAD has an id of 1
#UNK has id of 0

def create_vocab(filename, max_size=None, min_freq=1, savefile=None, specials = [UNK_TOK, PAD_TOK, SOS_TOK, EOS_TOK]):
    """
    Create a vocabulary object
    Args
        filename (str) : filename to induce vocab from
        max_size (int) : max size of the vocabular (None = Unbounded)
        min_freq (int) : the minimum times a word must appear to be 
        placed in the vocab
        savefile (str or None) : file to save vocab to (return it if None)
        specials (list) : list of special tokens 
    returns Vocab object
    """
    nlp = spacy.load('en')
    count = Counter()
    with open(filename, 'r') as fi:
        csv_file = csv.reader(fi)
        for line in csv_file:
            string = " ".join(line[2:])
            tokens = nlp.tokenizer(string)
#            for tok in line.replace(",", " ").split(" "):
#                count.update([tok.rstrip('\n').replace('"', '').replace('.','').replace('"', '').replace("!",'')])
            count.update([x.text for x in tokens])

    voc = Vocab(count, max_size=max_size, min_freq=min_freq, specials=specials)
    if savefile is not None:
        with open(savefile, 'wb') as fi:
            pickle.dump(voc, fi)
        return None
    else:
        return voc


def load_vocab(filename):
    #load vocab from json file
    with open(filename, 'rb') as fi:
        voc = pickle.load(fi)
    return voc

def sentiment_label_vocab():
    return Vocab(Counter([POS_LABEL, NEG_LABEL, NEU_LABEL]), specials=[])


class ExtendableField(ttdata.Field):
    'A field class that allows the vocab object to be passed in' 
    #This is to avoid having to calculate the vocab every time 
    #we want to run
    def __init__(self, vocab, *args, include_lengths=True, **kwargs):
        """
        Args    
            Same args as Field except
            vocab (torchtext Vocab) : vocab to init with
                set this to None to init later

            USEFUL ARGS:
            tokenize
            fix_length (int) : max size for any example, rest are padded to this (None is defualt, means no limit)
            include_lengths (bool) : Whether to return lengths with the batch output (for packing)
        """

        super(ExtendableField, self).__init__(*args, pad_token=PAD_TOK, batch_first=True, include_lengths=include_lengths,**kwargs)
        if vocab is not None:
            self.vocab = vocab
            self.vocab_created = True
        else:
            self.vocab_created = False

    def init_vocab(self, vocab):
        if not self.vocab_created:
            self.vocab = vocab
            self.vocab_created = True

    def build_vocab(self):
        raise NotImplementedError

    def numericalize(self, arr, device=None, train=True):
        """Turn a batch of examples that use this field into a Variable.

        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.

        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (-1 or None): Device to create the Variable's Tensor on.
                Use -1 for CPU and None for the currently active GPU device.
                Default: None.
            train (boolean): Whether the batch is for a training set.
                If False, the Variable will be created with volatile=True.
                Default: True.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.LongTensor(lengths)

        if self.use_vocab:
            if self.sequential:
                arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]
            else:
                arr = [self.vocab.stoi[x] for x in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab, train)
        else:
            if self.tensor_type not in self.tensor_types:
                raise ValueError(
                    "Specified Field tensor_type {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.tensor_type))
            numericalization_func = self.tensor_types[self.tensor_type]
            # It doesn't make sense to explictly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            if not self.sequential:
                arr = [numericalization_func(x) if isinstance(x, six.string_types)
                       else x for x in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None, train)

        arr = self.tensor_type(arr)
        if self.sequential and not self.batch_first:
            arr.t_()
        if device == -1:
            if self.sequential:
                arr = arr.contiguous()
        else:
            arr = arr.cuda(device)
            if self.include_lengths:
                lengths = lengths.cuda(device)
        #print("arr is {}".format(arr))
        if self.include_lengths:
            return arr, lengths
        return arr


class RocStoryBatches(ttdata.Iterator):

    def combine_story(self, batch, pad_id=1):
        """
        Convert a batch (output from iterating over iterator) into an output 
        of the form Tensor [num_sents, batch, seqlen], this can be input directly into model
        
        returns:
            batch (Tensor, [num_sents, batch, seq_len]) : Tensor of input ids for the embeddings lookup
            seq_lens (Tensor [num_sents, batch]) : Store the sequence lengths for each batch for packing
        """
        s1, s1_len = batch.sent_1 #s1 will be a Tensor [batch, seq_len], s1_len is Tensor of size [batch_size]
        s2, s2_len = batch.sent_2
        s3, s3_len = batch.sent_3
        s4, s4_len = batch.sent_4
        s5, s5_len = batch.sent_5
        seq_lens = torch.stack([s1_len, s2_len, s3_len, s4_len, s5_len], dim=0)
        
        max_len = torch.max(seq_lens)

        sents = [s1, s2, s3, s4, s5]

        for i in range(len(sents)):
            curr_len = sents[i].shape[1]
            pad_len = max_len - curr_len
            sents[i] = F.pad(sents[i], (0, pad_len), mode='constant', value=pad_id)

        sents = torch.stack(sents, dim=0)

        return sents, seq_lens

    def combine_sentiment_labels(self, batch, use_cuda=False):
        """
        Convert a sentiment targets and labels into an output 
        of the form Tensor [num_sents, batch], this can be input directly into model
        
        returns:
            (Tensor [num_sents, batch]) :
        """
        s1= batch.sent_1_lab #s1 is [batch_size]
        s2= batch.sent_2_lab
        s3= batch.sent_3_lab
        s4= batch.sent_4_lab
        s5= batch.sent_5_lab
        
        if use_cuda:
            targets = torch.stack([s1,s2,s3,s4,s5], dim=0).cuda()
        else:
            targets = torch.stack([s1,s2,s3,s4,s5], dim=0)
            
        return targets


    def convert_to_target(self, input, seq_lens):
        """
        Convert an input to target (ie just remove the BOS from the start
        Args:
            input (tensor, [num_sents, batch, seq_len])
            seq_lens (Tensor, [num_sents, batch])
        """
        num_sents = input.shape[0]
        targets = []
        for i in range(num_sents):
            targets.append(input[i, :, 1:])
        targets = torch.stack(targets, dim=0)
        return targets, seq_lens - 1


    def combine_story_cloze(self, batch, pad_id=1):

        s1, s1_len = batch.sent_1
        s2, s2_len = batch.sent_2
        s3, s3_len = batch.sent_3
        s4, s4_len = batch.sent_4
        s5, s5_len = batch.opt1
        s6, s6_len = batch.opt2
        tid = batch.tid
        seq_lens1 = torch.stack([s1_len, s2_len, s3_len, s4_len, s5_len], dim=0)
        seq_lens2 = torch.stack([s1_len, s2_len, s3_len, s4_len, s6_len], dim=0)

        max_len = torch.max(seq_lens1)
        sents1 = [s1, s2, s3, s4, s5]
        for i in range(len(sents1)):
            curr_len = sents1[i].shape[1]
            pad_len = max_len - curr_len
            sents1[i] = F.pad(sents1[i], (0, pad_len), mode='constant', value=pad_id)
        sents1 = torch.stack(sents1, dim=0)

        
        max_len = torch.max(seq_lens2)
        sents2 = [s1, s2, s3, s4, s6]
        for i in range(len(sents2)):
            curr_len = sents2[i].shape[1]
            pad_len = max_len - curr_len
            sents2[i] = F.pad(sents2[i], (0, pad_len), mode='constant', value=pad_id)
        sents2 = torch.stack(sents2, dim=0)

        return sents1, seq_lens1, sents2, seq_lens2, tid


class RocStoryClozeDataset(ttdata.Dataset):

    def __init__(self, path, vocab, preprocessed_examples=None, skip_header=True):

        if skip_header:
            print("Loading validation/test dataset. Skipping header.")
        else:
            print("NOT skipping header.")

        sent_1 = ExtendableField(vocab, init_token=SOS_TOK, eos_token=EOS_TOK, tokenize="spacy")
        sent_2 = ExtendableField(vocab, init_token=SOS_TOK, eos_token=EOS_TOK, tokenize="spacy")
        sent_3 = ExtendableField(vocab, init_token=SOS_TOK, eos_token=EOS_TOK, tokenize="spacy")
        sent_4 = ExtendableField(vocab, init_token=SOS_TOK, eos_token=EOS_TOK, tokenize="spacy")
        sent_5 = ExtendableField(vocab, init_token=SOS_TOK, eos_token=EOS_TOK, tokenize="spacy")
        sent_6 = ExtendableField(vocab, init_token=SOS_TOK, eos_token=EOS_TOK, tokenize="spacy")    
        tid = ttdata.RawField()

        fields = [('sent_1', sent_1), ('sent_2', sent_2),('sent_3', sent_3),('sent_4', sent_4),('opt1', sent_5), ('opt2', sent_6), ('tid', tid)]
        examples = []

        if preprocessed_examples is not None:
            super(RocStoryClozeDataset, self).__init__(preprocessed_examples, fields)
            
        else: 
            print("Loading RocStories (Validation/Testing) Set")

            with open(path, 'r') as f:
                csv_file = csv.reader(f)
                #Line format is id, sent1, sent2, sent3, sent4, option1, option2, target_id
                for i, line in enumerate(csv_file):
                    if i == 0:
                        continue
                     
                    _, s1, s2, s3, s4, opt1, opt2, tid = line
                         
                    examples.append(ttdata.Example.fromlist([s1, s2, s3, s4, opt1, opt2, tid], fields))

     
            super(RocStoryClozeDataset, self).__init__(examples, fields)

    
class RocStoryDataset(ttdata.Dataset):
    'CSV containing the full RocStory dataset, used for unsupervised training'

    def __init__(self, path, vocab, test=False, preprocessed_examples=None):

        """
        Args
            path (str) : Filename of RocStory CSV
            vocab (Torchtext Vocab object)
            test : Whether or not this is a test set or a training set (the csv is slightly different depending on it)
        """

        sent_1 = ExtendableField(vocab, init_token=SOS_TOK, eos_token=EOS_TOK, tokenize="spacy")
        sent_2 = ExtendableField(vocab, init_token=SOS_TOK, eos_token=EOS_TOK, tokenize="spacy")
        sent_3 = ExtendableField(vocab, init_token=SOS_TOK, eos_token=EOS_TOK, tokenize="spacy")
        sent_4 = ExtendableField(vocab, init_token=SOS_TOK, eos_token=EOS_TOK, tokenize="spacy")
        sent_5 = ExtendableField(vocab, init_token=SOS_TOK, eos_token=EOS_TOK, tokenize="spacy")
       
        fields = [('sent_1', sent_1), ('sent_2', sent_2),('sent_3', sent_3),('sent_4', sent_4),('sent_5', sent_5)]
        examples = []

        if preprocessed_examples is not None:
            super(RocStoryDataset, self).__init__(preprocessed_examples, fields)
            
        else:
            if not test:
                print("Loading RocStories (Training) Set")
            else:
                print("Loading RocStories (Validation/Testing) Set")

            with open(path, 'r') as f:
                csv_file = csv.reader(f)
                #Line format is id, title, sent1, sent2, sent3, sent4, sent5
                for i, line in enumerate(csv_file):
                    if i == 0:
                        print("SKIPPING THE HEADER")
                        continue
                    
                    if not test:
                        s1, s2, s3, s4, s5 = line[2:7]
                    else:
                        s1, s2, s3, s4, s5 = line[1:6]
                        
                    examples.append(ttdata.Example.fromlist([s1, s2, s3, s4, s5], fields))

     
            super(RocStoryDataset, self).__init__(examples, fields)


class RocStoryDatasetSentiment(ttdata.Dataset):
    'CSV containing the full RocStory dataset, with a sentiment tag for each sentence'

    def __init__(self, path, vocab, label_voc, test=False, preprocessed_examples=None):

        """
        Args
            path (str) : Filename of RocStory CSV
            vocab (Torchtext Vocab object)
            test : Whether or not this is a test set or a training set (the csv is slightly different depending on it)
        """

#        label_voc = Vocab(Counter([POS_LABEL, NEG_LABEL, NEU_LABEL]), specials=[])

        sent_1 = ExtendableField(vocab, init_token=SOS_TOK, eos_token=EOS_TOK, tokenize="spacy")
        sent_2 = ExtendableField(vocab, init_token=SOS_TOK, eos_token=EOS_TOK, tokenize="spacy")
        sent_3 = ExtendableField(vocab, init_token=SOS_TOK, eos_token=EOS_TOK, tokenize="spacy")
        sent_4 = ExtendableField(vocab, init_token=SOS_TOK, eos_token=EOS_TOK, tokenize="spacy")
        sent_5 = ExtendableField(vocab, init_token=SOS_TOK, eos_token=EOS_TOK, tokenize="spacy")

        sent_1_lab = ExtendableField(label_voc, include_lengths=False)
        sent_2_lab= ExtendableField(label_voc, include_lengths=False)
        sent_3_lab= ExtendableField(label_voc, include_lengths=False)
        sent_4_lab= ExtendableField(label_voc, include_lengths=False)
        sent_5_lab= ExtendableField(label_voc, include_lengths=False)

       
        fields = [('sent_1', sent_1), ('sent_2', sent_2),('sent_3', sent_3),('sent_4', sent_4),('sent_5', sent_5),
                  ('sent_1_lab', sent_1_lab), ('sent_2_lab', sent_2_lab),('sent_3_lab', sent_3_lab),('sent_4_lab', sent_4_lab),('sent_5_lab', sent_5_lab)]
        examples = []

        if preprocessed_examples is not None:
            super(RocStoryDatasetSentiment, self).__init__(preprocessed_examples, fields)
            
        else:
            if not test:
                print("Loading RocStories (Training) Set")
            else:
                print("Loading RocStories (Validation/Testing) Set")

            with open(path, 'r') as f:
                csv_file = csv.reader(f)
                #Line format is id, title, sent1, sent2, sent3, sent4, sent5, sent_1 lab, ...
                for i, line in enumerate(csv_file):
                    if not test:
                        sentences_labels = line[2:]
                    else:
                        sentences_labels = line[1:6] + line[7:]
                        
                    examples.append(ttdata.Example.fromlist(sentences_labels, fields))
     
            super(RocStoryDatasetSentiment, self).__init__(examples, fields)


def transform(output, dict):
    out = ""
    for i in output:
        out += " " + dict[i]
    return out


class S2SSentenceDataset(ttdata.Dataset):
    'Reads from 1 file and extracts src and tgt. Data set which has a single sentence per line'

    def __init__(self, path, input_vocab, output_vocab, skip_header=False):

        """
        Args
            path (str) : Filename of text file with dataset
            vocab (Torchtext Vocab object) 
        """
        if skip_header:
            print("Loading validation/testing dataset. Skipping header.")
        else:
            print("Loading Training dataset. NOT Skipping header.")
        text_field = ExtendableField(input_vocab)
        target_field = ExtendableField(output_vocab, tokenize="spacy", eos_token=EOS_TOK)
      
        fields = [('text', text_field), ('target', target_field)]
        examples = []
        with open(path, 'r') as f: 
            csv_file = csv.reader(f)
            if skip_header:
                print("SKIPPING THE HEADER")
                next(csv_file) # skipping the header in val file
            for line in csv_file: 
                text, target = line[-5:], line[2:7]
                text = " ".join(text).strip()
                target = " ".join(target).strip()
                #print(text, "|||", target)
                examples.append(ttdata.Example.fromlist([text, target], fields))
                #break
 
        super(S2SSentenceDataset, self).__init__(examples, fields)

