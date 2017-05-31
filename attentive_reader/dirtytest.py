import tensorflow as tf
from tensorflow.contrib import rnn
import time
import os
import pickle
from data_utility import batch_generator as bg


#datapath = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_example/mail"
datapath = '/home/dashmoment/workspace/text_summary/attentive_reader/dataset/summary_case_example/mail'
files = os.listdir(datapath)

filelist = []

for f in files:
    filelist = filelist + [os.path.join(datapath,f)]

gnerate_dql_pairs_folder(filepath, labeldict,label_type, qfile, lstm_step,q_lstm_step ,batchsize,embeddingfile,embeddingsize, dtype = 'NA'):
#text = open(filelist[0], "r").read()
#
#
#with open('w2v.pickle', 'rb') as handle:
#    wordvec = pickle.load(handle)
    

