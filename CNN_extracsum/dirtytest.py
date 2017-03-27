#text = "<t>i am one.</t><t>i am two.</t>"
#
#import  data_process 
#
#import re 
#
#pattern1 = re.compile(r'\<p>(.*?)\</p>')
#pattern2 = re.compile(r'\<b>(.*?)\</b>')
#
#replaced = re.sub(pattern1, r'\1', text)
#replaced = re.sub(pattern2, r'\1', replaced)
#
#
#
#res2 = data_process.filetagstriper("pos.txt")
#print(res2)
#
#
#xx,yy = data_process.load_data_and_labels_notag("pos.txt", "neg.txt")
#
#print("pos:{}".format(xx))
#print("neg:{}".format(yy))
#print(type(xx))
#
#xx,yy = data_process.load_data_and_labels("pos.txt", "neg.txt")
#print("npos:{}".format(xx))
#print("nneg:{}".format(yy))
#
#print(type(xx))

batches = data_process.batch_iter(list(zip(x_train, y_train)), 3, 10)

for batch in batches:
    packed = batch
    x3, y3= zip(*batch)
    break