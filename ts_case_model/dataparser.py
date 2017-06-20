import numpy as np
import re
import pickle
import os
import shutil
import statics
import sys
import operator

import collections

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

def create_data_label_path(dataset_path_list, dl_pair_path):
    
    if os.path.isfile(dl_pair_path): return statics.loadfrompickle(dl_pair_path)

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
                
                if type(filepath) is not list: filepath = [filepath]
                
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
                
    data_dict =   statics.savetopickle(dl_pair_path, data_dict)   
    return data_dict           
                    
def process_data_to_pickle(process_root, path_dict, wdict_path, ldict_path):
    
    
    if os.path.isfile(wdict_path):
        wdicts = statics.loadfrompickle(wdict_path)
    else:
        wdicts = {}
        
        
    if os.path.isfile(ldict_path):
        
        ldicts = statics.loadfrompickle(ldict_path)
    else:
        ldicts = {'model':{},'OS':{},'category':{}}
        
   
    
    count = 0

    for d in  path_dict:
                  
        count = count + 1
        sys.stdout.write("Data to pickle:{}/{}\n".format(count,  len(path_dict)))
        sys.stdout.flush()
        
        casefolder = os.path.join(process_root, d)
        
        if not os.path.isdir(casefolder):
            os.mkdir(casefolder)
        
        file_idx = 0 
        
        if type(path_dict[d]['context']) is not list: path_dict[d]['context'] = [path_dict[d]['context']]
        
        for clist in  path_dict[d]['context']:
            
            for c in clist:
                             
                savepath = os.path.join(casefolder, str(file_idx)+'.pickle')
                
                if os.path.isfile(savepath): continue
                
                if not os.path.isfile(savepath):
                
                    stripe = slipt_doc_by_space(clist[c])               
                    wdicts = collect_dict(stripe, wdict_path, wdicts)
                    
                    with open(savepath, 'wb') as f:
                         pickle.dump(stripe, f, protocol=pickle.HIGHEST_PROTOCOL)
           
            file_idx = file_idx + 1
         
        lsavepath = os.path.join(casefolder, 'label.pickle')
        if os.path.isfile(lsavepath): continue
        
        if not os.path.isfile(lsavepath):
            with open(lsavepath, 'wb') as lf:
                
                labels = slipt_label(path_dict[d]['label'])
                pickle.dump(labels, lf, protocol=pickle.HIGHEST_PROTOCOL)
                ldicts = collect_dict(labels, ldict_path, ldicts)
                
                
def collect_dict(data, dict_path,  wdicts={'model':{},'OS':{},'category':{}}):
    
    if type(data) is dict:
      
        lmodel = data['model'][0]
        los = data['OS'][0]
        lcat = data['category'][0]
        
       
        if lmodel not in wdicts['model']: wdicts['model'][lmodel] = len(wdicts['model'])
        if los not in wdicts['OS']: wdicts['OS'][los] = len(wdicts['OS'])
        if lcat not in wdicts['category']: wdicts['category'][lcat] = len(wdicts['category'])
                      
       
    else:
    
        for w in data:
            
            if w in wdicts: 
                wdicts[w] = wdicts[w] + 1
            else:
                wdicts[w] = 1
                      
    statics.savetopickle(dict_path, wdicts)        
    
    return wdicts


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
process_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2'

#dataset_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/'
#process_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/summary_processed'
wdict_path = os.path.join(process_root, 'wdict.pickle')
ldict_path = os.path.join(process_root, 'ldict.pickle')


final_ldict_path = os.path.join(process_root, 'final_ldict.pickle')
final_wdict_path = os.path.join(process_root, 'final_wdict20k.pickle')
final_rwdict_path = os.path.join(process_root, 'final_rwdict20k.pickle')
dl_pair_path = os.path.join(process_root, 'dl_path_pair.pickle')

#dataset_path_list = [os.path.join(dataset_root, 'tier1'), os.path.join(dataset_root, 'tier2')]  
dataset_path_list = [os.path.join(dataset_root, 'summary_data')]    
path_dict = create_data_label_path(dataset_path_list, dl_pair_path)           



#dataset_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/toy_test/rawdata/'
#process_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/toy_test/process'
#wdict_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/toy_test/process/wdict.pickle'
#ldict_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/toy_test/process/ldict.pickle'

#dataset_root = '/home/dashmoment/dataset/toy_test'
#process_root = '/dataset/ts_case_process'
#wdict_path = '/dataset/ts_case_process/wdict.pickle'
#ldict_path = '/dataset/ts_case_process/ldict.pickle'
#path_dict = create_data_label_path([dataset_root])    

#process_data_to_pickle(process_root, path_dict, wdict_path, ldict_path)



def getChinese(context):
   # context = context.decode("utf-8") # convert context from str to unicode
    filtrate = re.compile(u'[^\u4E00-\u9FA5]') # non-Chinese unicode range
    context = filtrate.sub(r'', context) # remove all non-Chinese letters
    #context = context.encode("utf-8") # convert unicode back to str
    return context




def create_wdict_ldict(Nword, wdict_path, ldict_path, final_wdict_path, final_ldict_path):
    
    qwords = ["what", "is", " model",  "OS", "category"]
    
    wdict = statics.loadfrompickle(wdict_path)
    ldict = statics.loadfrompickle(ldict_path)
    sorted_wdict = sorted(wdict.items(), key=operator.itemgetter(1))
    
    r_wdict = {}
    pure_dict = {}
    
    
    label_type = ['OS', 'category', 'model']
    
    r_ldict = {}
    r_ldict[0] = 'UNKL'
    pure_ldict = {}
    pure_ldict['UNKL'] = 0
    
    count = 1
    for lt in label_type:
        
        
         for l in ldict[lt]:
             
             if len(l) > 1  and l not in pure_ldict:
                 r_ldict[count] = l
                 pure_ldict[l] = len(pure_ldict)
                 count = count + 1
                 
             elif len(l) > 1:
                 
            
                 l = lt+l
                 r_ldict[count] = l    
                 pure_ldict[l] = len(pure_ldict)
                 count = count + 1
                 print(l)
                 
             
            
    
    
    r_wdict[0] = 'UNK'
    pure_dict['UNK'] = 0
    
    idx = 1
    count = 1
    while idx < Nword + 1:
        
#        print("Create_dict:{}/{}".format(count, len(sorted_wdict)))
        
       
        
        if len(getChinese(sorted_wdict[-count][0])) == 0: 
            r_wdict[idx] = sorted_wdict[-count][0]
            pure_dict[r_wdict[idx]] = idx
            idx = idx + 1
        
        count = count + 1
    
    
    for i in range(1,len(r_ldict)):
        
        if r_ldict[i] not in pure_dict:
            pure_dict[r_ldict[i]] = len(r_wdict)
            r_wdict[len(r_wdict)] = r_ldict[i]
        
    for i in range(len(qwords)):
        
        if qwords[i] not in pure_dict:
            pure_dict[qwords[i]] = len(r_wdict)
            r_wdict[len(r_wdict)] = qwords[i]

    statics.savetopickle(final_wdict_path, pure_dict)  
    statics.savetopickle(final_ldict_path, pure_ldict) 
    statics.savetopickle(final_rwdict_path, r_wdict)  
          
    return pure_dict, pure_ldict, r_wdict

Nword = 20000
pure_dict, pure_ldict, r_wdict = create_wdict_ldict(Nword,  wdict_path, ldict_path,  final_wdict_path, final_ldict_path)






