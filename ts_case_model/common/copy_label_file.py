import numpy as np
import pickle
import tensorflow as tf
import os
import shutil

def savetopickle(filepath, obj):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
        
def loadfrompickle(filepath):
    with open(filepath, 'rb') as f:
        
        obj = pickle.load(f)
    f.close()    
    return obj


src_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/'
output_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset_summary/raw'
sub_folder = ['tier1', 'tier2']

for folder in sub_folder:
    
    count = 0
      
    source_path = os.path.join(src_path, folder)
    folder_list = os.listdir(source_path)
    
    for f in folder_list:
        count = count + 1
        print("Process:{}/{}".format(count, len(folder_list)))
        
        out_path = os.path.join(output_path, f)
        
        if os.path.isdir(out_path):
            
            l_file = os.path.join(source_path, f+ '/labels.txt')
            shutil.copy2(l_file, out_path)
        else:
            shutil.copytree(os.path.join(source_path,f), output_path)
            
            