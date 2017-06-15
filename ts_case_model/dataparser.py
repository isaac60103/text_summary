import numpy as np
import re
import pickle
import os
import shutil
import statics
import sys

#------------Context File Paser---------


def slipt_doc_by_space(filename, re_reg = '<a href.*?</a>|www.*|[^-_\w]'):
    
    text = open(filename, "r").read()
    
    cleantext = slipt_words_by_space(text, re_reg)
    
    return cleantext


def slipt_words_by_space(text, re_reg  = '<a href.*?</a>|www.*|[^-_\w]'):
    
    r = re.compile(re_reg)
    text= r.sub(" ", text)
    
    
    re_reg = '[-_]'
    r = re.compile(re_reg)
    text= r.sub("", text)
    
    
    splited = re.split('\s', text)
    
    cleantext = list(filter(None, splited))
    cleantext = list(filter(lambda x: len(x)> 3, cleantext))
    cleantext = list(filter(lambda x: len(x)<20, cleantext))
    cleantext = list(filter(lambda x: not x.isdigit() , cleantext))
    
    return cleantext

    
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
    label_dict['subject'] =  strip_label_content(splited[3], False)
    
    r = re.compile('Description:')
    
    des_sub = [r.sub("", x) for x in splited[4:]]
    
    description = ''
    
    for idx in range(len(des_sub)): description = description + des_sub[idx]
    
    description = slipt_words_by_space(description)
    

    label_dict['description'] =  description

    
    return label_dict

def strip_label_content(splited_text, split=True):

    tmp = []
    text = re.split(':', splited_text)
  
    for idx in range(1,len(text)):
        tmp.append(text[idx])
      
    
    if split == True: 
        re_reg = '[-_]'
        r = re.compile(re_reg)
        tmp[0]= r.sub("", tmp[0])
        tmp[0] = tmp[0].replace(" ","")
    
    return tmp


# Create a dictionary contain all mails and label in each case
#   dict['case name'] = {mail_context_path, label_file_path}

def create_data_label_path(dataset_path_list):

    data_dict = {}
    
    count = 0
    
    for d in dataset_path_list:
              
        lsdir = os.listdir(d)
        
        for folder in lsdir:
            
            count = count + 1
            sys.stdout.write("Create path dict:{}/{}\n".format(count,  len(lsdir)))
            sys.stdout.flush()
            
            folderpath = os.path.join(d, folder)
            filelist = os.listdir(folderpath)
            data_dict[folder] = {}
            
            if len(filelist) > 1:
                filelist.remove('labels.txt')
                filepath = [{x:os.path.join(folderpath, x)} for x in filelist]
                
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
                    
def process_data_to_pickle(process_root, path_dict, wdict_path, ldict_path):
    
    wdicts = {}
    ldicts = {}

    for d in  path_dict:  
        
        casefolder = os.path.join(process_root, d)
        
        if not os.path.isdir(casefolder):
            os.mkdir(casefolder)
        
        file_idx = 0 
        
        for clist in  path_dict[d]['context']:
            
            for c in clist:
                
                         
                savepath = os.path.join(casefolder, str(file_idx)+'.pickle')
                
                if not os.path.isfile(savepath):
                
                    stripe = slipt_doc_by_space(clist[c])
                    wdicts = collect_dict(stripe, wdict_path, wdicts)
                    
                    with open(savepath, 'wb') as f:
                         pickle.dump(stripe, f, protocol=pickle.HIGHEST_PROTOCOL)
           
            file_idx = file_idx + 1
         
        lsavepath = os.path.join(casefolder, 'label.pickle')
        
        if not os.path.isfile(lsavepath):
            with open(lsavepath, 'wb') as lf:
                
                labels = slipt_label(path_dict[d]['label'])
                pickle.dump(labels, lf, protocol=pickle.HIGHEST_PROTOCOL)
                ldicts = collect_dict(labels, ldict_path, ldicts)
                
                
def collect_dict(data, dict_path,  wdicts={}):
    
    for w in data:
        
        if w in wdicts: 
            wdicts[w] = wdicts[w] + 1
        else:
            wdicts[w] = 1
                  
    statics.savetopickle(wdicts, dict_path)        
    return wdicts


#def build_dataset(words, n_words):
#  """Process raw inputs into a dataset."""
#  count = [['UNK', -1]]
#  count.extend(collections.Counter(words).most_common(n_words - 1))
#  dictionary = dict()
#  for word, _ in count:
#    dictionary[word] = len(dictionary)
#  data = list()
#  unk_count = 0
#  for word in words:
#    if word in dictionary:
#      index = dictionary[word]
#    else:
#      index = 0  # dictionary['UNK']
#      unk_count += 1
#    data.append(index)
#  count[0][1] = unk_count
#  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
#  
#  data = list(filter(lambda a: a != 0, data))
#  return data, count, dictionary, reversed_dictionary       
                           
#dataset_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset'
#dataset_path_list = [os.path.join(dataset_root, 'tier1')]    
#path_dict = create_data_label_path(dataset_path_list)           
#process_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed'
#
#rpath = os.path.join('/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed/5002000000GPRxB', '0.pickle')
#with open(rpath, 'rb') as f:
#        label_dict_res = pickle.load(f)


dataset_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/toy_test'
process_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/process_new'
 path_dict, wdict_path, ldict_path

path_dict = create_data_label_path([dataset_root])    

process_data_to_pickle

#case_list = os.listdir(dataset_root)
#case_path = os.path.join(dataset_root, case_list[39])
#
#mail_list = os.listdir(case_path)
#
#mail_path = os.path.join(case_path, mail_list[0])
#
#
#clean_text = slipt_doc_by_space(mail_path)
#
#label_path = os.path.join(case_path, 'labels.txt')
#
#
#label = slipt_label(label_path)








