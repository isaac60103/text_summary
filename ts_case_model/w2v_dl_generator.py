import numpy as np
import os
import statics
import collections
import random



final_ldict_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2/final_ldict.pickle'
final_wdict_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2/final_wdict20k.pickle'
process_data_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2'
word_pool_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/w2vec/words_pool.pickle'


wdict = statics.loadfrompickle(final_wdict_path)
ldict = statics.loadfrompickle(final_ldict_path)


#-----------------Create all words-------------------------------

src_list = os.listdir(process_data_root)

dl_pair = []


if os.path.isfile(word_pool_path): words = statics.loadfrompickle(word_pool_path)
else: words = []

for i in range(6000, len(src_list)):

    
    print("Process: {}/{}".format(i, len(src_list)))
    
    dir_path = os.path.join(process_data_root, src_list[i])
    
    if os.path.isdir(dir_path) : file_path = os.listdir(dir_path)
    else: continue
      
    
    for f in file_path:
            
        if f != 'label.pickle':
            path = os.path.join(dir_path, f)
            words = words + statics.loadfrompickle(path)
            
    if i%1000 == 0: statics.savetopickle(word_pool_path, words)  


#-----------------Encode words by dict-------------------------------      
#code_words = []
#for i in words:
#    code_words.append(wdict[i])
    
#data = code_words
#data_index = 0
#
#def generate_batch(batch_size, num_skips, skip_window):
#  global data_index
#  assert batch_size % num_skips == 0
#  assert num_skips <= 2 * skip_window
#  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
#  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
#  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
#  buffer = collections.deque(maxlen=span)
#  if data_index + span > len(data):
#    data_index = 0
#  buffer.extend(data[data_index:data_index + span])
#  data_index += span
#  for i in range(batch_size // num_skips):
#    target = skip_window  # target label at the center of the buffer
#    targets_to_avoid = [skip_window]
#    for j in range(num_skips):
#      while target in targets_to_avoid:
#        target = random.randint(0, span - 1)
#      targets_to_avoid.append(target)
#      batch[i * num_skips + j] = buffer[skip_window]
#      labels[i * num_skips + j, 0] = buffer[target]
#    if data_index == len(data):
#      buffer[:] = data[:span]
#      data_index = span
#    else:
#      buffer.append(data[data_index])
#      data_index += 1
#  # Backtrack a little bit to avoid skipping words in the end of a batch
#  data_index = (data_index + len(data) - span) % len(data)
#  return batch, labels   



#batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)




