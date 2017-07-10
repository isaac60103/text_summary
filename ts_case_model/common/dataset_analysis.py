import tensorflow as tf
import csv
import os
import sys

sys.path.append('../')
import common.statics as stat




def encode_labels(label_dict, label):
    
    label_type = ["OS", "category", "model"]
    
   
    label_enc = {}
    
    for ltype in label_type:
              
        
        if label[ltype][0] in ldict: 
            label_enc[ltype] = label[ltype][0]
           
            
        else:
            
            label_enc[ltype] = "UNK_"+ltype
           
           
#    encode_label = np.reshape(encode_label, (-1, LABEL_SIZE))
    
    return label_enc
    


final_wdict_path ='/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2/final_wdict20k.pickle'
final_ldict_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2/final_ldict_general.pickle'
encode_dict = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/w2vec/w2v_dict100k.pickle'

wdict = stat.loadfrompickle(final_wdict_path)
ldict = stat.loadfrompickle(final_ldict_path)

src_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2'
dir_list = os.listdir(src_path)

statics_result = []


count = 0

for idx in range(0,len(dir_list)):
    
    print("Process: {}/{}".format(idx, len(dir_list)))
    

    dir_path = os.path.join(src_path, dir_list[idx])
    
    if os.path.isdir(dir_path):
        file_list = os.listdir(dir_path)
        
         
        words = []
        
        for fid in range(len(file_list)) :
                
                if file_list[fid] != 'label.pickle':
                    fpath = os.path.join(dir_path, file_list[fid])
                    words = words + stat.loadfrompickle(fpath)
                    
                else:
                    fpath = os.path.join(dir_path, 'label.pickle')
                    label = stat.loadfrompickle(fpath)
                    
       
            
    count = count + 1
    
     
    enc_label = encode_labels(ldict, label)
    
    print(dir_list[idx])
    
    statics_dict = {}
    
    statics_dict['case'] = dir_list[idx]
    statics_dict['words'] = len(words)
    statics_dict['OS'] =enc_label['OS']
    statics_dict['category'] = enc_label['category'] 
    statics_dict['model'] = enc_label['model']
   
    statics_result.append(statics_dict)
           
 
keys = statics_result[0].keys()

with open('test.csv', 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, keys, delimiter=',')
    dict_writer.writerows(statics_result)





        