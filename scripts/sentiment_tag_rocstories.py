import numpy as np
import math
import csv
import sys
import pickle
import argparse
from collections import defaultdict, Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_sentiment(vader, sent):
    pol = vader.polarity_scores(sent)
    compound = pol['compound']
    if compound >= 0.2:
        tag='POS'
    elif compound <= -1*0.2:
        tag='NEG'
    else:
        tag='NEU'

    return tag


def sentiment_tag(args, test=False):
    compound_thresh = 0.2
    out = []
    vader = SentimentIntensityAnalyzer()
    with open(args.input_file, 'r') as f:
        csv_file = csv.reader(f)
        #Line format is id, title, sent1, sent2, sent3, sent4, sent5
        header = next(csv_file)
        print("Length of header {}".format(len(header)))
        print("SKIPPING THE HEADER")
        for i, line in enumerate(csv_file):
            #if i == 0:
                #print("SKIPPING THE HEADER")
                #continue
            
            if not test:
                sentences= line[2:]
            else:
                sentences = line[1:6]
            
            tags = []
            for sent in sentences:
                tags.append(get_sentiment(vader, sent)) 
            out.append(line + tags)

    with open(args.output_file, 'w') as f:
        writer = csv.writer(f) 
        for i in out:
            writer.writerow(i)

def test_sentiment_tag(args):
    vader = SentimentIntensityAnalyzer()
    with open(args.input_file) as fg:
        golden = fg.readlines()

    golden = [line.strip().split() for line in golden] # list of lists 
    length = len(golden)
    print("Total gold labels {}".format(length))

    output = [] # list of lists
    story_tags = []
    with open(args.output_file) as fo:
        for line in fo: 
            line = line.rstrip("\n")
            if line: 
                #print("Appending to story")
                story_tags.append(get_sentiment(vader, line))
            else: # add a blank line for this to work
                assert len(story_tags) == 5, "Must be 5 sentences"
                #print("Finished with one story")
                output.append(story_tags)
                story_tags = []
 
    assert len(golden) == len(output) * 5, "{} and {} Must be same length".format(len(golden), len(output)*5) 

    accuracy = sum([1 if g == o else 0 for g, o in zip(golden, output)])
    print("Accuracy {:.4f} || {}/{}".format(accuracy/length, accuracy, length))
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SLDS')
    parser.add_argument('--input_file', type=str, help="Input file/reference file when evaluating") 
    parser.add_argument('--output_file', type=str, help="Output file") 
    parser.add_argument('--acc', action='store_true', help="To evaluate")       
    args = parser.parse_args()
 
    if args.acc:
        print("Calculating ACCURACY")
        test_sentiment_tag(args) 
    else:
        sentiment_tag(args)

