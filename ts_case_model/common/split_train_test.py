import os
import shutil

root_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/processed_v2'
train_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/train'
test_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ts_cases_dataset/test'

src_files = os.listdir(root_path)

portion = int(len(src_files)*0.8)
train_files = src_files[:portion]
test_files = src_files[portion:]

for fid in range(len(src_files)):
    src_path = os.path.join(root_path, src_files[fid])
    
    
    
    if os.path.isdir(src_path):
        print("Process {}/{}".format(fid, len(src_files)))
    
        if src_files[fid] in train_files:
        
            dst_path  = os.path.join(train_path, src_files[fid])
        else:
            dst_path  = os.path.join(test_path, src_files[fid])
            
        shutil.copytree(src_path, dst_path, False, None)
    
    
print("Number of train tasks:", len(os.listdir(train_path)))
print("Number of test tasks:", len(os.listdir(test_path)))