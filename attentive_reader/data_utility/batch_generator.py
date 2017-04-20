import numpy as np
import re
import pickle
import data_utility.dataparser as parser


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
         
     padsize = targetlen - len(words)
     
     assert padsize > 0, "Content length is longer than target length. Please slice your contents or extend target length."
     
     pad = np.zeros([1,128])
     pad[0] = wordvec['UNK']
     
     for i in range(padsize):
         words = np.vstack((words,pad))
         
     return words

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

#filename = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_example/mail/02s0O00000rOMxF.txt"
#qfilename = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_example/question/q1.txt"
#lfilename = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_example/label/labels.txt"#

#labels = genrate_label_fromfile(lfilename)
##print(labels)#

#dql = gnerate_dql_pairs_fromfile(filename, qfilename, lfilename, '../w2v.pickle',128, 200,20)
#print(dql)