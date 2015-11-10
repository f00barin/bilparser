import numpy as np
import operator
from collections import defaultdict
from itertools import permutations


trainhml = [tuple(w.strip().split()) for w in open('penn-wsj-deps/02/head_mod_label_list.txt')]
trainhmlset = set(trainhml)
trainhset = set([w[0] for w in trainhml]) 
trainmset = set([w[0] for w in trainhml])
trainmset = set([w[1] for w in trainhml])
trainhmset = set([(w[0], w[1]) for w in trainhml])

hmlmat = np.load('vbfeatures.pkl')
f = open('penn-wsj-deps/03/ptb.val.original.conll')
data_list = []
current_index = 0
word_list = []
pos_list = []
edge_set = {}
ed_set = []
current_index = 0
for line in f:
    line = line[:-1]
    if line != '': 
        current_index += 1
        entity = line.split()
        word_list.append(entity[0])
        pos_list.append(entity[1])
        edge_set[(int(entity[2]), current_index)] = entity[3]
    else:
        if word_list != []:
            ed_set = []
            word_list = ['__ROOT__'] + word_list
            for (h,m) in edge_set.iterkeys():
                ed_set.append((word_list[h], word_list[m], edge_set[(h, m)]))
            data_list.append((ed_set, word_list))
        word_list = []
        pos_list = []
        edge_set = {}
        ed_set = []
        current_index = 0
f.close() 


def sortdict(d):
    return sorted(d.items(), key=operator.itemgetter(1))

def getnbest(d, n):
    return sortdict(d)[-n:]

def getnbestelement(d, n):
    retlist = []
    fulllist = getnbest(d,n)
    for i in xrange(n):
        retlist.append(fulllist[i][0])
    return set(retlist)

def extractrelindo(data, hmlmat, trainhset, trainmset, trainhmset, trainhmlset, fulllist=[]):
    labels = ['AMOD','DEP','NMOD','OBJ','PMOD','PRD','SBAR','SUB','VC','VMOD']
    for instance in data:
        goldlist = instance[0]
        wlist = instance[1]
        
        ldict = dict.fromkeys(labels, {})
        tmparr = []
        for (h,m) in permutations(wlist, 2):
            for l in labels:
                if (h,m) in hmlmat[l]:
                    ldict[l].update({(h,m): hmlmat[l][(h,m)]})
        for (h, m, l) in goldlist:
            if l in labels:
                if h in trainhset:
                    tmparr.append(1)
                else:
                    tmparr.append(0)
                if m in trainmset:
                    tmparr.append(1)
                else:
                    tmparr.append(0)
                if (h,m) in trainhmset:
                    tmparr.append(1)
                else:
                    tmparr.append(0)
                if (h,m,l) in trainhmlset:
                    tmparr.append(1)
                else:
                    tmparr.append(0)
                if (h,m) in hmlmat[l]:
                    if hmlmat[l][(h,m)] >= 0: 
                        tmparr.append(1)
                    else:
                        tmparr.append(0)
                    if hmlmat[l][(h,m)] < 0:
                        tmparr.append(1)
                    else:
                        tmparr.append(0)
                    if (h,m) in getnbestelement(ldict[l],1):
                        tmparr.append(1)
                    else:
                        tmparr.append(0) 
                    if len(ldict[l].keys()) > 3:
                        try:
                            if (h,m) in getnbestelement(ldict[l],3):
                                tmparr.append(1)
                            else:
                                tmparr.append(0)
                        except:
                            print ldict[l]
                    else: 
                        tmparr.append(0)
                else:
                    tmparr += [-1, -1, -1, -1]
            else:
                tmparr = [-1, -1, -1, -1, -1, -1, -1, -1]
            fulllist.append(tmparr)
            tmparr= []
    return fulllist


arr = extractrelindo(data_list, hmlmat, trainhset, trainmset, trainhmset, trainhmlset, fulllist=[])

import cPickle
cPickle.dump(arr, open('fullarr.pkl', 'wb'))
