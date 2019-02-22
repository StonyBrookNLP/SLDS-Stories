import pickle
import numpy as np
import re
import time
# import torch


class wordIndex(object):
    '''
    Class for creating the dictionary
    Have a common vocabulary for the line, context and the label words
    '''
    def __init__(self):
        #Initializations
        #start the count with 1 so that 0 is left for zero padding
        self.count = 1
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = {}
        ###### For tokenization
        # self.tokens = 0
        # self.token = 0
        # self.ids = torch.LongTensor(self.tokens)
        #############
        #assign a dummy shape to the embeddings weight matrix - will change!
        self.weights_matrix = np.zeros((10, 2))
        ##Remove this from the init
        self.data_sentences = list()
        self.data_char_contexts = list()
        self.data_labels = list()
        # self.data_sentences_ids = list()
        # self.data_char_contexts_ids = list()
        # self.data_labels_ids = list()

    def add_word(self,word):
        if word not in self.word2idx:
            # print word
            self.word2idx[word] = self.count
            # print self.word2idx[word]
            self.word_count[word] = 1
            self.count += 1
        else:
            self.word_count[word] += 1

    def add_text(self,text):
        #tokenize text and add it to the dictionary
        words = text.split()
        # self.tokens += len(words)
        for word in words:
            self.add_word(word)

    def add_to_dictionary(self, dataset):
        for line_combine in dataset:
            self.add_text(line_combine)

    def map_id_to_word(self):
        for word in self.word2idx:
            self.idx2word[self.word2idx[word]] = word

def get_dataset(input_data):
    data_sentences = list()
    data_char_contexts = list()
    data_labels = list()
    #read training data
    for story_id in input_data:
        story_data = input_data[story_id]

        for points in story_data:
            # print(points)
            label = normalizeString(points.keys()[0])
            data_labels.append(label)

            sentence = normalizeString(points.values()[0][0])
            data_sentences.append(sentence)
            
            context = normalizeString(points.values()[0][1])
            data_char_contexts.append(context)
    
    # print len(data_labels)
    # print len(data_sentences)
    # print len(data_char_contexts)

    return data_sentences, data_char_contexts, data_labels

def normalizeString(s):
    # if len(s) > 0:
    s = s.lower().strip()
    s = re.sub(r"<br />",r" ",s)
    s = re.sub(r'(\W)(?=\1)', '', s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)

    return s

# def limitDict(limit,obj):
#     dict1 = sorted(obj.word_count.items(),key = lambda t : t[1], reverse = True)
#     count = 0
#     for x,y in dict1 :
#         if count >= limit-1 :
#             obj.word2idx[x] = limit
#         else :
#             obj.word2idx[x] = count 

#         count+=1

def getGloveEmbeddings(obj):
    matrix_len = len(obj.word2idx)
    #adding a 1 to account for the additional unknown word

    obj.weights_matrix = np.zeros((matrix_len + 2, EMB_DIM))
    #dictionary to hold the word embeddings
    words_found = 0
    # spec_char_table = maketrans()
    for word in obj.word2idx:
        try:
            obj.weights_matrix[obj.word2idx[word]] = glove[word]
            words_found += 1 

        except Exception as e:
            # print e
            # print("Word with no pre-trained embedding vectors:", word)
            obj.weights_matrix[obj.word2idx[word]] = np.random.normal(EMB_DIM, )
    #assign randomly initialized vector for the words not seen in the training data
    #assign as the last index
    # print EMB_DIM
    # print np.random.normal(size=(EMB_DIM,))
    # print
    #adding a randomly initialized embedding vector for the 
    obj.weights_matrix[matrix_len+1] = np.random.normal(size=(EMB_DIM,))

    #remove the first row which contains only zeros
    obj.weights_matrix = obj.weights_matrix[1:]
    # print obj.weights_matrix["UNKNOWN"]

    # print obj.weights_matrix

def tokenize_id(obj, dataset, input_type):
    
    dataset_ids = []
    len_vocab = len(obj.word2idx)

    for line in dataset:
        line_list = line.split()
        line_list_ids = list()

        # if "empty" in line_list:
            # print line_list
            # print


        for word in line_list:
            try:
                # if input_type == "labels":
                #     # print word
                #     # print obj.word2idx[word]
                #     # print
                #     pass
                if word == "empty" and (len(line_list) == 1):
                    print "Line list:"
                    print line_list
                    print "When empty context:"
                    print word
                    print obj.word2idx[word]
                    print
                    line_list_ids.append(0)

                else:
                    line_list_ids.append(obj.word2idx[word])
            except Exception, e:
                #unseen word in the test data
                #instead of the keyword "UNKNOWN", put this as the integer = size of the vocabulary (so last index)
                line_list_ids.append(len_vocab)
            
        dataset_ids.append(line_list_ids)
    return dataset_ids

def convertTextToID(obj, dataset):
    # print ""
    data_sentences_ids = tokenize_id(obj, dataset["sentence"], "sentence")
    data_char_contexts_ids = tokenize_id(obj, dataset["char_contexts"], "char_contexts")
    data_labels_ids = tokenize_id(obj, dataset["labels"], "labels")
    
    # print data_labels_ids[:10]
    # print data_sentences_ids[:10]
    # print data_char_contexts_ids[:10]

    return data_sentences_ids, data_char_contexts_ids, data_labels_ids

def output_dataset(obj):

    # if split == "training":
    ######################################outputting the training data#####################################################################
    #output the sentences
    pickle.dump(train_data_sentences, open(data_dir + "training_sentences.p", "wb"))
    #output the character contexts
    # print obj.data_char_contexts[:10]
    pickle.dump(train_data_char_contexts, open(data_dir + "training_char_contexts.p", "wb"))
    #output the labels
    # print obj.data_labels[:10]
    pickle.dump(train_data_labels, open(data_dir + "training_labels.p", "wb"))

    # print obj.data_sentences_ids[:10]
    pickle.dump(train_data_sentences_ids, open(data_dir + "training_sentences_id.p", "wb"))

    # print obj.data_char_contexts_ids[:10]
    pickle.dump(train_data_char_contexts_ids, open(data_dir + "training_char_contexts_id.p", "wb"))

    # print obj.data_labels_ids[:10]
    pickle.dump(train_data_labels_ids, open(data_dir + "training_labels_id.p", "wb"))

    #output the word2idx dictionary
    # print obj.data_labels[:10]
    pickle.dump(obj.word2idx, open(data_dir + "training_word2idx.p", "wb"))
    #output the idx2word list
    # print obj.idx2word
    pickle.dump(obj.idx2word, open(data_dir + "training_idx2word.p", "wb"))
    #output the glove embeddings        
    pickle.dump(obj.weights_matrix, open(data_dir + "training_glove_matrix.p", "wb"))

    # else:
    #output the sentences
    pickle.dump(dev_data_sentences, open(data_dir + "validation_sentences.p", "wb"))
    #output the character contexts
    pickle.dump(dev_data_char_contexts, open(data_dir + "validation_char_contexts.p", "wb"))
    #output the labels
    pickle.dump(dev_data_labels, open(data_dir + "validation_labels.p", "wb"))

    # print obj.data_sentences_ids[:10]
    pickle.dump(dev_data_sentences_ids, open(data_dir + "validation_sentences_id.p", "wb"))

    # print obj.data_char_contexts_ids[:10]
    pickle.dump(dev_data_char_contexts_ids, open(data_dir + "validation_char_contexts_id.p", "wb"))

    # print obj.data_labels_ids[:10]
    pickle.dump(dev_data_labels_ids, open(data_dir + "validation_labels_id.p", "wb"))        

    #output the sentences
    pickle.dump(test_data_sentences, open(data_dir + "test_sentences.p", "wb"))
    #output the character contexts
    pickle.dump(test_data_char_contexts, open(data_dir + "test_char_contexts.p", "wb"))
    #output the labels
    pickle.dump(test_data_labels, open(data_dir + "test_labels.p", "wb"))

    # print obj.data_sentences_ids[:10]
    pickle.dump(test_data_sentences_ids, open(data_dir + "test_sentences_id.p", "wb"))

    # print obj.data_char_contexts_ids[:10]
    pickle.dump(test_data_char_contexts_ids, open(data_dir + "test_char_contexts_id.p", "wb"))

    # print obj.data_labels_ids[:10]
    pickle.dump(test_data_labels_ids, open(data_dir + "test_labels_id.p", "wb"))    


#Hyperparameters for the model
EMB_DIM = 100
vocabLimit = 50000
max_sequence_len = 500

##################################################################################################
#### Read in the glove vectors ##################################################################

data_dir = '../../../data_for_code_new_entities/'
#using the above objects, create a dictionary that given a word returns its vector
glove = pickle.load(open('../../../data_for_code/glove.6B.100d.dat', 'rb'))
words = pickle.load(open('../../../data_for_code/glove.6B.100_words.pkl', 'rb'))
# word2idx = pickle.load(open('../../../data_for_code/glove.6B.50_idx.pkl', 'rb'))

#input data
training_data = pickle.load(open(data_dir + "training_data.p", "rb"))
val_data = pickle.load(open(data_dir + "validation_data.p", "rb"))
test_data = pickle.load(open(data_dir + "test_data.p", "rb"))

# training_data_sentences, training_data_char_contexts, training_data_labels = get_dataset(training_data, obj1)
# val_data_sentences, val_data_char_contexts, val_data_labels = get_dataset(val_data, obj2)

#get the sentences, contexts and labels in three separate lists while also adding to the dictionary
#pass the corresponding object for the dataset involved
train_data_sentences, train_data_char_contexts, train_data_labels = get_dataset(training_data)
dev_data_sentences, dev_data_char_contexts, dev_data_labels = get_dataset(val_data)
test_data_sentences, test_data_char_contexts, test_data_labels = get_dataset(test_data)

train_combine = train_data_sentences + train_data_char_contexts + train_data_labels
# dev_combine = dev_data_sentences + dev_data_char_contexts + dev_data_labels

#Object for getting word indices - training
obj = wordIndex()

obj.add_to_dictionary(train_combine)

print obj.word2idx["anger"]
print obj.word2idx["surprise"]
print obj.word2idx["fear"]
print obj.word2idx["anticipation"]
print obj.word2idx["sadness"]
print obj.word2idx["none"]
print obj.word2idx["disgust"]
print obj.word2idx["trust"]
print obj.word2idx["joy"]
print obj.word2idx["none"]

# print obj.word2idx
obj.map_id_to_word()

#limit the words by their frequency
# limitDict(vocabLimit, obj)   #in the training data

#####Replace the words with their ids##############################
print "Training data:"
train_data_sentences_ids, train_data_char_contexts_ids, train_data_labels_ids = convertTextToID(obj, {"sentence" : train_data_sentences, "char_contexts": train_data_char_contexts, "labels" : train_data_labels})
print "Validation data:"
dev_data_sentences_ids, dev_data_char_contexts_ids, dev_data_labels_ids = convertTextToID(obj, {"sentence" : dev_data_sentences, "char_contexts" : dev_data_char_contexts, "labels" : dev_data_labels})
print "Test data:"
test_data_sentences_ids, test_data_char_contexts_ids, test_data_labels_ids = convertTextToID(obj, {"sentence" : test_data_sentences, "char_contexts" : test_data_char_contexts, "labels" : test_data_labels})


# print

####Add glove vectors for the words in the vocabulary###################################
getGloveEmbeddings(obj)

print obj.weights_matrix
# getGloveEmbeddings.keys()
#Output the training and validation data to be fed in the model
output_dataset(obj)







