import numpy as np
import re
import pickle

def slipt_text_by_space(text):
    
    r = re.compile('[^-_a-z0-9A-Z]')
    text= r.sub(" ", text)
    splited = re.split('\s', text)
    cleantext = list(filter(None, splited))
    
    return cleantext

def slipt_doc_by_space(filename):
    
    text = open(filename, "r").read()
    r = re.compile('[^-_a-z0-9A-Z]')
    
    text= r.sub(" ", text)
    
    splited = re.split('\s', text)
    
    cleantext = list(filter(None, splited))
    
    return cleantext


def create_vocdicts_files(filelist):
    
    vocab = []
    
    for f in filelist:
        vocab = vocab + slipt_doc_by_space(f)
    
    
    return vocab

def mapword2dict(words, dictionary):
    
    word2dict = words

    for idx in range(len(words)):
        
        try:
            word2dict[idx] = dictionary[words[idx]]
        except:
            word2dict[idx] = 0
    
    word2dict = list(filter(lambda a: a != 0, word2dict))
    
    return word2dict
    

def wordembededbycontents(words, embeddingsize ,embeddingfile):
    
    with open(embeddingfile, 'rb') as handle:
        wordvec = pickle.load(handle)
    
    embedding = np.zeros([len(words),embeddingsize])
    
    for idx in range(len(words)):
        try:
            embedding[idx] = wordvec[words[idx]]
        except:
            embedding[idx] = wordvec['UNK']
            
    return embedding
    




#filename = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_example/mail/02s0O00000rOMxF.txt"
#qfilename = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_example/question/q1.txt"
#words = slipt_doc_by_space(filename)
#q = slipt_doc_by_space(qfilename)#

#en = wordembededbycontents(words, 'w2v.pickle')
#qen  = wordembededbycontents(q, 'w2v.pickle')
#h = padvec(210, en, 'w2v.pickle')#

#lfilename = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_example/label/labels.txt"
#lc = open(lfilename, "r").read()
#labels = genrate_label(lc)

#with open('w2v.pickle', 'rb') as handle:
#    wordvec = pickle.load(handle)
#    
#padsize = 200 - len(words)
#pad = np.zeros([1,128])
#pad[0] = wordvec['UNK']
#
#for i in range(padsize):
#    en = np.vstack((en,pad))


        
























