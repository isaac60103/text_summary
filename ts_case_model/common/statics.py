import numpy as np
import os
import pickle
import collections


def savetopickle(filepath, obj):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
        
def loadfrompickle(filepath):
    with open(filepath, 'rb') as f:
        
        obj = pickle.load(f)
    f.close()    
    return obj

def calc_tfidf(process_root):

    all_words = [] 
    
     
    idx = 0
    total_document = 0
    
    for case in os.listdir(process_root):
        
        print("calc_tf process:{}/{}".format(idx, len(os.listdir(process_root))))
        idx = idx + 1
        
        casepath = os.path.join(process_root, case)
        
        
        for context in os.listdir(casepath):
            if context != 'label.pickle':
                total_document = total_document + 1
                contextpath = os.path.join(casepath, context)
                all_words  = all_words + loadfrompickle(contextpath)
    
    count = []
    count.extend(collections.Counter(all_words).most_common(len(all_words)))
    
    tfidf_score = {}
    
    for w in count: tfidf_score[w[0]] = [w[1]/len(all_words), 0, 0]
    
    savetopickle('tf_score.pickle', tfidf_score)
        
    idx = 0
        
    for k in tfidf_score:
        
        print("calc_idf process:{}/{}".format(idx, len(tfidf_score)))
        idx= idx + 1
        
        for case in os.listdir(process_root):
            
            casepath = os.path.join(process_root, case)
            
            for context in os.listdir(casepath):
                if context != 'label.pickle':
                    
                    contextpath = os.path.join(casepath, context)
                    c = loadfrompickle(contextpath)
                    if k in c: tfidf_score[k][1] = tfidf_score[k][1] + 1
    
    for k in tfidf_score:
        tfidf_score[k][1] = np.log(total_document/(tfidf_score[k][1]+1))
        tfidf_score[k][2] = tfidf_score[k][0] * tfidf_score[k][1]
                
                
    return total_document, tfidf_score
                        
def calc_label(process_root, threshold = 3):
    
    label_type = ['model', 'OS', 'category']
    
    label_dict = {}
    
    for i in label_type: label_dict[i] = []
        
    idx = 0
    
    for case in os.listdir(process_root):
        
        idx= idx + 1
        print("Process:{}/{}".format(idx, len(os.listdir(process_root))))
        
        
        casepath = os.path.join(process_root, case)
        labelpath = os.path.join(casepath, 'label.pickle')
        
        with open(labelpath, 'rb') as f:
            label = pickle.load(f)
        
        for i in label_type: 
            label[i][0] = label[i][0].replace(" ","")
            label_dict[i] = label_dict[i] + label[i]
            
    label_static = {}
    for i in label_type: label_static[i] = []
    
 
    for i in label_type: 
        
        res = collections.Counter(label_dict[i]).most_common(len(label_dict[i]))
        label_static[i] = [x for x in res if x[1] > threshold]
    
    return label_static


#Input : return value of calc_label
def create_label(process_root='' ,pickfile = '' ,ltype = 'all'):
    
    if pickfile != '':
        label_static = loadfrompickle(pickfile)
    else:
        label_static = calc_label(process_root, 3)
        
    
    label_index = 0
    labellist = []
    
    if ltype != 'all':
        total_class = len(label_static[ltype])
        
        for i in range(total_class):
            zero_label = np.zeros(total_class + 1, np.float32)
            zero_label[label_index] = 1
            labellist.append([label_static[ltype][i][0], zero_label])
            label_index = label_index + 1
        
        zero_label = np.zeros(total_class + 1, np.float32)
        zero_label[label_index] = 1
        labellist.append([ltype+'_UNK', zero_label])
        
        
    else:
        
        total_class = 0
        for i in label_static: total_class = total_class + len(label_static[i])
        
        
        for ctype in label_static:
            for i in range(len(label_static[ctype])):
            
                zero_label = np.zeros(total_class + 3, np.float32)
                zero_label[label_index] = 1
                    
                labellist.append([label_static[ctype][i][0], zero_label])
                
                label_index = label_index + 1
            zero_label = np.zeros(total_class + 3, np.float32)
            zero_label[label_index] = 1
            labellist.append([ctype+'_UNK', zero_label])
            
            label_index = label_index + 1
        
    return  labellist

#process_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed'
#context_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/toy_test'
#labellist = create_label(process_root = process_root)
#
#tfidf_score = calc_tfidf(process_root)
#savetopickle('tfidfscore.pickle', tfidf_score)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    