import torch 
import torch.nn as nn
import numpy as np
import math
import csv
import spacy
import pickle
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torchtext.data as ttdata
import torchtext.datasets as ttdatasets
from torchtext.vocab import Vocab
from collections import defaultdict, Counter

from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Convert the roc stories csv file into a rocstories csv file with sentiment tags (for each sentence) at end
def sentiment_tag(args, test=False):
    compound_thresh = 0.2
    out = []
    vader = SentimentIntensityAnalyzer()
    with open(args.input_file, 'r') as f:
        csv_file = csv.reader(f)
        #Line format is id, title, sent1, sent2, sent3, sent4, sent5
        for i, line in enumerate(csv_file):
            if i == 0:
                continue
            
            if not test:
                sentences= line[2:]
            else:
                sentences = line[1:6]
            
            tags = []
            for sent in sentences:
                pol = vader.polarity_scores(sent)
                compound = pol['compound']
                if compound >= 0.2:
                    tag='POS'
                elif compound <= -1*0.2:
                    tag='NEG'
                else:
                    tag='NEU'

                tags.append(tag)
            out.append(line + tags)

    with open(args.output_file, 'w') as f:
        writer = csv.writer(f)
        for i in out:
            writer.writerow(i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SLDS')
    parser.add_argument('--input_file', type=str) #Pickle file with a list of examples of RocStoriesDataset type
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--vocab', type=str, help='the vocabulary pickle file')
       
    args = parser.parse_args()

   
    sentiment_tag(args)

