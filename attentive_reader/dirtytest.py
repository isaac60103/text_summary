import numpy as np
import re
import math
import itertools
from collections import Counter
import collections
import random
import tensorflow as tf
import data_utility as du
import os
import sys
import pickle


datapath = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/summary_case_example/mail"
files = os.listdir(datapath)

filelist = []

for f in files:
    filelist = filelist + [os.path.join(datapath,f)]


text = open(filelist[0], "r").read()


with open('w2v.pickle', 'rb') as handle:
    wordvec = pickle.load(handle)
    



