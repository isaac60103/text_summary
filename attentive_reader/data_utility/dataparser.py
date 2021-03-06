import numpy as np
import re
import pickle
import os
import shutil


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



#------------Context Label File Paser---------


def slipt_text_by_space(text, re_reg = '[^-_a-z0-9A-Z]'):
    
    r = re.compile(re_reg)
    text= r.sub(" ", text)
    splited = re.split('\s', text)
    cleantext = list(filter(None, splited))
    
    return cleantext

def slipt_doc_by_space(filename, re_reg = '[^-_a-z0-9A-Z]'):
    
    text = open(filename, "r").read()
    r = re.compile(re_reg)
    
    text= r.sub(" ", text)
    
    splited = re.split('\s', text)
    
    cleantext = list(filter(None, splited))
    
    return cleantext

def create_vocdicts_files(filelist, re_reg = '[^-_a-z0-9A-Z]'):
    
    vocab = []
    idx = 0
    total = len(filelist)
    for f in filelist:
        idx = idx + 1
        print("Process Files:{}/{}".format(idx, total))
        vocab = vocab + slipt_doc_by_space(f,re_reg)
    
    return vocab
   
#---------------------Label File Paser-----------------
#----------------Lables will be parsed as dict---------

def slipt_label(filename):
    
    text = open(filename, "r").read()
    #print(text)
    
    splited = re.split('\n', text)
    #print(splited)
    
    label_dict = {}
    
    label_dict['model'] =  strip_label_content(splited[0])
    label_dict['category'] =  strip_label_content(splited[1])
    label_dict['OS'] =  strip_label_content(splited[2])
    label_dict['subject'] =  strip_label_content(splited[3])
    
    r = re.compile('Description:')
    
    des = [r.sub(" ", x) for x in splited[4:]]
    r = re.compile('[^-_a-z0-9A-Z\']')
    des = [r.sub(" ", x) for x in des]
    des = list(filter(lambda x: len(x)> 10, des))
    
    resdes = []
    
    for i in des:
         resdes = resdes + re.split(' ', i)
        
    
    label_dict['description'] =  list(filter(None, resdes))

    #print(label_dict)

    
    return label_dict

def strip_label_content(splited_text, split=True):

    tmp = []
    if split==True: text = re.split(':', splited_text)
    else: text = splited_text

    for idx in range(1,len(text)):
        tmp.append(text[idx])

    return tmp

def create_vocdicts_label(filelist):

    label_dict = {}
    label_dict['model'] =  []
    label_dict['category'] =  []
    label_dict['OS'] =  []
    label_dict['subject'] =  []
    label_dict['description'] =  []

    idx = 0
    for f in filelist:
        idx = idx + 1
        #print("Process Files:{}/{}".format(idx, total))
        label = slipt_label(f)
        label_dict['model'] = label_dict['model'] + label['model']
        label_dict['category'] = label_dict['category'] + label['category']
        label_dict['OS'] = label_dict['OS'] + label['OS']
        label_dict['subject'] = label_dict['subject'] + label['subject']
        label_dict['description'] = label_dict['description'] + label['description']


    return label_dict





# Create a dictionary contain all mails and label in each case
#   dict['case name'] = {mail_context_path, label_file_path}
def create_data_label_path(dataset_path_list):

    data_dict = {}
    
    
    for d in dataset_path_list:
        lsdir = os.listdir(d)
        
        for folder in lsdir:
            
            folderpath = os.path.join(d, folder)
            filelist = os.listdir(folderpath)
            data_dict[folder] = {}
            
            if len(filelist) > 1:
                filelist.remove('labels.txt')
                filepath = [os.path.join(folderpath, x) for x in filelist]
                
                data_dict[folder]['context'] = filepath 
            else:
                data_dict[folder]['context'] = {}
            
            label_path = os.path.join(folderpath, 'labels.txt')
            
            
            
            text = open(label_path, "r").read()
            
            if data_dict[folder]['context']=={}:
                
                if len(text) == 0:
                    
                    shutil.rmtree(folderpath)
                else:
                    sl = slipt_label(label_path)
                   
                    if len(sl['description']) != 0:
                        
                                        
                        context_path = os.path.join(folderpath, 'context.txt') 
                        
                        with open(context_path, 'w') as f:
                            for line in sl['description']:
                                f.write(line+" ")
                            f.close
                        
                        data_dict[folder]['label'] = label_path
                        data_dict[folder]['context'] = context_path
                        
                    else:
                        del data_dict[folder]
            else:
                data_dict[folder]['label'] = label_path
                
                
    return data_dict           
                    

dataset_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset'
dataset_path_list = [os.path.join(dataset_root, 'tier1'), os.path.join(dataset_root, 'tier2')]    
path_dict = create_data_label_path(dataset_path_list)           

#    for x in data_dict:
#        
#        if data_dict[x]['context']=={}:
#            
#            text = open(data_dict[x]['label'], "r").read()
#            if len(text) == 0:
#                p = os.path.join(d, x)
#                shutil.rmtree(p)
#            else:
#                sl = slipt_label(data_dict[x]['label'])
#                if sl['description']
                
















