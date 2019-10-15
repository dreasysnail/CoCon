import numpy as np 
import sys
import pdb

fh = open(sys.argv[1],'rb')
new = open(sys.argv[2],'w')
for l in fh.readlines():
    items = l.rstrip('x').rstrip().split('\t')
    for idx, t in enumerate(items):
        if idx % 7 == 0:
            new.write(t + '\t')
    new.write('\n')

fh.close()
new.close()
