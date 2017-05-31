import numpy as np
import re
import pickle
import os
import shutil
import collections

#------------Context File Paser---------


def slipt_doc_by_space(filename, re_reg = '[^-_a-z0-9A-Z\']'):
    
    text = open(filename, "r").read()
    r = re.compile(re_reg)
    
    text= r.sub(" ", text)
    
    splited = re.split('\s', text)
    
    cleantext = list(filter(None, splited))
    
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
                    
def process_data_to_pickle(process_root, path_dict):

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
                    
                    with open(savepath, 'wb') as f:
                         pickle.dump(stripe, f, protocol=pickle.HIGHEST_PROTOCOL)
           
            file_idx = file_idx + 1
         
        lsavepath = os.path.join(casefolder, 'label.pickle')
        
        if not os.path.isfile(lsavepath):
            with open(lsavepath, 'wb') as lf:
                pickle.dump(slipt_label(path_dict[d]['label']), lf, protocol=pickle.HIGHEST_PROTOCOL)


def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  
  data = list(filter(lambda a: a != 0, data))
  return data, count, dictionary, reversed_dictionary       
                           
dataset_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset'
dataset_path_list = [os.path.join(dataset_root, 'tier1'), os.path.join(dataset_root, 'tier2')]    
path_dict = create_data_label_path(dataset_path_list)           
process_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed'

rpath = os.path.join('/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed/5002000000GPRxB', '0.pickle')
with open(rpath, 'rb') as f:
        label_dict_res = pickle.load(f)

















