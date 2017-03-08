text = "<t>i am one.</t><t>i am two.</t>"

import  data_process 

import re 

pattern1 = re.compile(r'\<p>(.*?)\</p>')
pattern2 = re.compile(r'\<b>(.*?)\</b>')

replaced = re.sub(pattern1, r'\1', text)
replaced = re.sub(pattern2, r'\1', replaced)



res2 = data_process.filetagstriper("test.txt")
print(res2)


xx,yy = data_process.load_data_and_labels_notag("test.txt", "neg.txt")

print("pos:{}".format(xx))
print("neg:{}".format(yy))