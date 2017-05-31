import numpy as np
import re
import pickle
#import data_utility.dataparser as parser
import dataparser as parser
import os


pattern = ['Product Model (TS Confirm)',
              'TS Case Category',
              'OS',
              'Subject',
          ]

qpattern = { 'what is the model name\n':'Product Model (TS Confirm)'}


def genrate_label_fromfile(filename):


  text = open(filename, "r").read()
  labels = genrate_label(text)
 
  return labels

def genrate_label(text):

  r = re.compile(':')
  labels={}

  splitdes = text.split('Description:\n',1)
  labels['Description'] = splitdes[1]

  splited = re.split('\n', splitdes[0])

  for c in splited:

    for p in pattern:
      if c.find(p) != -1:
        tmp = c.split(p,1)[1]
        tmp = r.sub("", tmp)
        labels[p] = tmp

  return labels


def padvec(targetlen, words, embeddingfile):
  
     with open(embeddingfile, 'rb') as handle:
         wordvec = pickle.load(handle)
         
     divide_size = len(words)//targetlen
     divide_list = []
     for i in range(divide_size):
         divide_list = divide_list + [targetlen + targetlen*i]
     
     split_words = np.split(words, divide_list)
    
     
     for i in range(len(split_words)):
         if len(split_words[i]) < targetlen:
         
             padsize = targetlen - len(split_words[i])        
             assert padsize > 0, "Content length is longer than target length. Please slice your contents or extend target length."
             
             pad = np.zeros([1,1024])
             pad[0] = wordvec['UNK']
             
             for s in range(padsize):
                 split_words[i] = np.vstack([split_words[i],pad])
         
     return split_words

def gnerate_dql_pairs_fromfile(filelist, questionfile ,labelfile , embeddingfile, embeddingsize, contentslength, querylength):

    data = np.empty((0,embeddingsize))
    dql_pairs = {}

    assert type(filelist) is list, "Only accept list of Email files."
    
    for f in filelist:

        text = parser.slipt_doc_by_space(f)    
        data = np.vstack((data, parser.wordembededbycontents(text,embeddingsize,embeddingfile)))

    data = padvec(contentslength, data, embeddingfile)
    dql_pairs['contents'] = data

    q_text = parser.slipt_doc_by_space(questionfile)
    query = parser.wordembededbycontents(q_text,embeddingsize,embeddingfile)
    query = padvec(querylength, query, embeddingfile)
    dql_pairs['query'] = query

    qtext = open(questionfile, "r").read()
    ltext = open(labelfile, "r").read()
    label = genrate_label(ltext)
    qp = qpattern[qtext]
    dql_pairs['label'] = label[qp]


    return dql_pairs

def gen_label_dict(filepath, ltype):
    
    label_dict = {}
    
    subdir = os.listdir(filepath)
    label_files = []
    
    for folder in subdir:
         folderpath = os.path.join(filepath, folder) 
         files = os.path.join(folderpath, 'labels.txt') 
         label_files = label_files + [files]
         print("process folder: {}".format(folder))
         
    labels = parser.create_vocdicts_label(label_files)
    
    clean_labels = list(filter(None, labels[ltype])) # Delete Empty label
    clean_labels = list(set(clean_labels)) #Delte Duplicate Label
    
    idx = 1
    
    for l in clean_labels:
        label_dict[l] = idx
        idx = idx + 1
    
    return label_dict
    

def gen_embeded_data(filelist, embeddingfile, embeddingsize):
    
    data = np.empty((0,embeddingsize))
    
    for f in filelist:
        text = parser.slipt_doc_by_space(f)  
        data = np.vstack((data, parser.wordembededbycontents(text,embeddingsize,embeddingfile)))
        
    return data

def gnerate_dql_pairs_folder(filepath, labeldict,label_type, qfile, lstm_step,q_lstm_step ,embeddingfile,embeddingsize, dtype = 'NA', pad=True):
  
    
    batch_pack = {}
    batch_pack['data'] = []
    batch_pack['label'] = []
    batch_pack['question'] = []    
    
#    batch_list = []  
    subdir = os.listdir(filepath)
    train_size = int(0.9*len(subdir))

    if dtype =='train':
      subdir = subdir[:train_size]
    elif dtype == 'test':
      subdir = subdir[train_size:]
    else:
      subdir = subdir

    print(dtype)
    print(subdir)

    
#    batch_list = [batch_list + [random.choice(subdir)] for i in range(batchsize)]
    
    idx = 0
    for folder in subdir:
        idx = idx + 1
        print('Progress:{}/{}'.format(idx, len(subdir)))
        
        folderpath = os.path.join(filepath, folder) 
        files = os.listdir(folderpath)
        files.remove('labels.txt')
        label_file =  os.path.join(folderpath, 'labels.txt')
        
        #-------Encode data & Stack Multiple Mails--------
        for i in range(len(files)):
            files[i] = os.path.join(folderpath, files[i] ) 
            
        onedata = gen_embeded_data(files,embeddingfile,embeddingsize)
        
        if pad== True:
            paddata = padvec(lstm_step, onedata, embeddingfile)
        else:
            paddata = onedata
    
        
        #-------Encode Label by Dict------------
        labels = parser.create_vocdicts_label([label_file])
        if labels[label_type][0] in labeldict:
            en_label = labeldict[labels[label_type][0]]
        else:
            en_label = 0
        
        #-------Encode Question---------
        
        question = gen_embeded_data([qfile],embeddingfile,embeddingsize)
        
        if pad== True:
            question = padvec(q_lstm_step, question, embeddingfile)
        
        batch_pack['data'] =  batch_pack['data'] + [paddata]
        batch_pack['label'] = batch_pack['label'] + [en_label]
        batch_pack['question'] = batch_pack['question'] + [question]

    return batch_pack


def gne_dql_feeddict(filepath, labeldict,label_type, qfile, lstm_step,q_lstm_step ,batchsize,embeddingfile,embeddingsize, dtype):
    
    batch_dict = gnerate_dql_pairs_folder(filepath, labeldict,label_type, qfile, lstm_step,q_lstm_step ,embeddingfile,embeddingsize,dtype)

    print("Get Batch")
    
    batch_pack = {}
    
    Nenity = len(labeldict) + 1
   
    batch_pack['data'] = np.empty([0,lstm_step, embeddingsize])
    batch_pack['label'] = np.empty([0,Nenity])
    batch_pack['question'] = np.empty([0,q_lstm_step, embeddingsize])
    
    idx = 0
    
    for i in range(len(batch_dict['data'])):


        idx = idx + 1
        print('Progress:{}/{}'.format(idx,len(batch_dict['data'])))
        
        for data in batch_dict['data'][i]:
            
            d = data.reshape([1,lstm_step, embeddingsize])
            batch_pack['data'] = np.append(batch_pack['data'],d, axis=0)
            
            q =  batch_dict['question'][i][0].reshape([1,q_lstm_step, embeddingsize])
            batch_pack['question'] = np.append(batch_pack['question'],q, axis=0)
        
            index = batch_dict['label'][i]       
            tmpl = np.zeros([1,Nenity])
            tmpl[0][index] = 1
            
            batch_pack['label'] = np.append(batch_pack['label'],tmpl, axis=0)
            
            
    return batch_pack
        

def randombatch(batch_dict, batchsize):

  total_size = len(batch_dict['label'])
  batch_list = np.random.randint(total_size, size=batchsize)

  data = [batch_dict['data'][idx] for idx in batch_list]
  label = [batch_dict['label'][idx] for idx in batch_list]
  question = [batch_dict['question'][idx] for idx in batch_list]

  return data, label, question



#datapath = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_example/result"
##lf = gen_label_dict("/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_example/result", 'model')
#embeddingfile = '../w2v.pickle'
#embeddingsize = 1024
#labeldict = '../model_dict.pickle'
#qfile = '/home/dashmoment/workspace/text_summary/attentive_reader/dataset/summary_case_example/question/q1/q1.txt'
#question = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_example/question/q1/q1.txt'
#filelist = gne_dql_feeddict(datapath,model_dict,'model',question,500,20,32,embeddingfile,128)
#    
#folderpath = '/home/dashmoment/workspace/text_summary/attentive_reader/dataset/summary_case_example/mail/case1'
#
#files = os.listdir(folderpath)
#files.remove('labels.txt')
#label_file =  os.path.join(folderpath, 'labels.txt')
#
#
#for i in range(len(files)):
#    files[i] = os.path.join(folderpath, files[i] )
#
#context = []       
#for f in files:
#    text = parser.slipt_doc_by_space(f)  
#    context = context + text
#onedata = gen_embeded_data(files,embeddingfile,embeddingsize)
#paddata = padvec(500, onedata, embeddingfile)
#
#folderpath = '/home/dashmoment/workspace/text_summary/attentive_reader/dataset/summary_case_example/mail/'
#batch = gnerate_dql_pairs_folder(folderpath, labeldict,'model', qfile, 500,100 ,embeddingfile,embeddingsize, 'NA', False)



#pad = padvec(20, filelist['data'][0],embeddingfile)
#d = gen_embeded_data(filelist,'../w2v.pickle',128)

#filename = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_example/mail/02s0O00000rOMxF.txt"
#qfilename = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_example/question/q1.txt"
#lfilename = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_example/label/labels.txt"#

#labels = genrate_label_fromfile(lfilename)
##print(labels)#

#dql = gnerate_dql_pairs_fromfile(filename, qfilename, lfilename, '../w2v.pickle',128, 200,20)