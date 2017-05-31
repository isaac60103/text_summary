import pickle
import os

import model_utility as mut
import data_utility.batch_generator as bg


def getlabel_dict(dataconfig):
    
    datapath = dataconfig['srcpath']
    label_dict = dataconfig['label_dict']
    label_type = dataconfig['label_type']
    
    if os.path.isfile(label_dict):
        print('Load Model Dict')
        with open(label_dict, 'rb') as handle:
            label_dict_res = pickle.load(handle)
        
    else:
        print('Create Model Dict and save')
        label_dict_res = bg.gen_label_dict(datapath, label_type)
        with open(label_dict, 'wb') as handle:
            pickle.dump(label_dict_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    print('Finish Load dict')
    
    return label_dict_res
    
def gnerate_dql(dataconfig, modelconfig, label_dict_path,dtype):
    
    
    subdir = os.listdir(dataconfig['srcpath'])
    train_size = int(0.9*len(subdir))

    if dtype =='train':
      subdir = subdir[:train_size]
    elif dtype == 'test':
      subdir = subdir[train_size:]
    else:
      subdir = subdir

    idx = 0
    fileidx = 1
    for folder in subdir:

        folderpath = os.path.join(dataconfig['srcpath'], folder) 
        idx = idx + 1
        
        print('Progress:{}/{}'.format(idx, len(subdir)))

        end_idx = bg.gnerate_data_pickle(dataconfig,folderpath,modelconfig,fileidx)
        bg.gnerate_label_pickle(dataconfig, folderpath, label_dict_path,fileidx, end_idx)
        bg.gnerate_question_pickle(dataconfig,modelconfig,fileidx, end_idx)
        fileidx = end_idx
    

def preparetraning(dataconfig, test_config, modelconfig, label_dict_path = {'OS':'os_dict.pickle', 'model':'model_dict.pickle','category':'category_dict.pickle'}):

    getlabel_dict(dataconfig)
    
    if(len(os.listdir(dataconfig['data'])) > 0
        and len(os.listdir(dataconfig[dataconfig['label_type']])) > 0
        and len(os.listdir(dataconfig['question'])) > 0):
        print("Training Data Exist")
    else:
        
        gnerate_dql(dataconfig, modelconfig, label_dict_path, 'train')
        assert len(os.listdir(dataconfig['data'])) == len(os.listdir(dataconfig[dataconfig['label_type']])), "Dataset create fail"
        assert len(os.listdir(dataconfig[dataconfig['label_type']])) == len(os.listdir(dataconfig['question'])), "Dataset create fail"

    if(len(os.listdir(test_config['data'])) > 0
        and len(os.listdir(test_config[dataconfig['label_type']])) > 0
        and len(os.listdir(test_config['question'])) > 0):

        print("Test Data Exist")
    else:
        gnerate_dql(test_config, modelconfig, label_dict_path, 'test')
        assert len(os.listdir(test_config['data'])) == len(os.listdir(test_config[dataconfig['label_type']])), "Dataset create fail"
        assert len(os.listdir(test_config[dataconfig['label_type']])) == len(os.listdir(test_config['question'])), "Dataset create fail"
    
    
    print('Well Prepared, start training')
    
    
    