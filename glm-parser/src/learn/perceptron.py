# -*- coding: utf-8 -*-
import sqlite3
import numpy as np
from scipy.io import mmread
from glob import iglob
from itertools import permutations

#import os.path
import shelve
import logging
logging.basicConfig(filename='glm_parser.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

class PerceptronLearner():

    def __init__(self, w_vector, max_iter=1):
        logging.debug("Initialize PerceptronLearner ... ")
        self.w_vector = w_vector
        self.max_iter = max_iter
        self.sentence = 0
        return

    def sequential_learn(self, f_argmax, data_pool=None, max_iter=-1, d_filename=None, dump_freq = 1):
        if max_iter <= 0:
            max_iter = self.max_iter

        logging.debug("Starting sequantial train...")
#
#        s = shelve.open('biltraining.db', flag='c', writeback=True)
#        headreps = {}
#        modreps = {}
#        for section in self.section_list:
#            data_path_with_section = self.reps + ("%02d/" % (section, ))
#            for file_name in iglob(data_path_with_section + '*heads*'):
#                headreps = {w.strip().split()[0]:np.array(list(map(float,
#                                                                        w.strip().split()[1:]))) for
#                             w in open(file_name)}
#            for file_name in iglob(data_path_with_section + '*mods*'):
#                modreps = {w.strip().split()[0]:np.array(list(map(float,
#                                                              w.strip().split()[1:])))
#                                for w in open(file_name)}
#        if headreps != {} and modreps != {}:
#            headmat = np.zeros((len(headreps), 300))
#            row = 0
#            headlist = []
#            for head, vector in headreps.items():
#                headmat[row] = vector
#                headlist.append(head)
#                row += 1
#
#            modmat = np.zeros((len(modreps), 300))
#            row = 0
#            modlist = []
#            print len(modreps)
#            for mod, vector in modreps.items():
#                modmat[row] = vector
#                modlist.append(mod)
#                row += 1
#
#            labels = ['AMOD','DEP','NMOD','OBJ','P','PMOD','PRD','SBAR','SUB','VC','VMOD']
#
#            for section in self.section_list:
#                data_path_with_section = self.reps + ("%02d/" % (section, ))
#                for label in labels:
#                    for file_name in iglob(data_path_with_section+'*'+label+'*'):
#                        wmat = np.array(mmread(file_name))
#                        scoremat = headmat.dot(wmat.dot(modmat.T))
#                        for hi in range(len(headlist)):
#                            for mi in range(len(modlist)):
#                                if label in s:
#                                    s[label][(headlist[hi], modlist[mi])] = \
#                                        (str(5, 0, headlist[hi], modlist[mi],
#                                            label), scoremat[hi, mi])
#                                else:
#                                    s[label] = {(headlist[hi], modlist[mi]): \
#                                         (str(5, 0, headlist[hi], modlist[mi],
#                                              \
#                                              label), scoremat[hi, mi])}
#                    s.sync()
#        s.close()
#

        for i in range(max_iter):
            logging.debug("Iteration: %d" % i)
            logging.debug("Data size: %d" % len(data_pool.data_list))

            s = shelve.open('learndataset.db', flag='c', writeback=True)
#            db = sqlite3.connect('validation_sql.db')
#            cur = db.cursor()
#            cur.execute('''CREATE TABLE features (sentno integer,  head
##                        integer, mod integer, feats text)''')
##            db.commit()
            while data_pool.has_next_data():
                self.sentence += 1
#                cur = db.cursor()
                sentno = str(self.sentence)
                data_instance = data_pool.get_next_data()
                print self.sentence
                for h, m in permutations(range(len(data_instance.word_list)),2):
##                    cur.execute("INSERT INTO features VALUES (?,?,?,?)",
##                               (self.sentence, h, m,
##                                '###join###'.join(data_instance.get_local_vector(h,m,
##                                                                                 self.sentence)[0])))
##                db.commit()
#
                    if sentno in s:
                        s[sentno][(h,m)] = data_instance.get_local_vector(h,m,
                                                                          self.sentence)[0]
                    else:
                        s[sentno] = {(h,m):
                                     data_instance.get_local_vector(h,m,
                                                                    self.sentence)[0]}
                s.sync()
            s.close()
#            db.close()
#                    print h,m
#                    cur = db.cursor()
#                    cur.execute("INSERT INTO features VALUES (?,?,?,?)",
#                               (self.sentence, h, m, '###join###'.join(data_instance.get_local_vector(h,m)[0])))
#                    db.commit()
#
#                for h,m in hmlist:
#                    print 'this is %s, %s' % (h,m), data_instance.get_local_vector(h,m)


#                gold_global_vector = data_instance.gold_global_vector
##                print gold_global_vector.keys()
#                current_global_vector = f_argmax(data_instance, self.sentence)
##                print current_global_vector.keys()
#                self.update_weight(current_global_vector, gold_global_vector)
#
#            data_pool.reset_index()
#            if d_filename is not None:
#                if i % dump_freq == 0 or i == max_iter - 1:
#                    self.w_vector.dump(d_filename + "_Iter_%d.db"%i)
#

    def update_weight(self, current_global_vector, gold_global_vector):
        # otherwise, the gold_global_vector will change because of the change in weights
        for key in gold_global_vector.keys():
            self.w_vector.data_dict[key].iadd(gold_global_vector[key].feature_dict)

#        print 'Gold ', gold_global_vector
#        print 'Current ', current_global_vector

        for key in current_global_vector.iterkeys():
            self.w_vector.data_dict[key].iaddc(current_global_vector[key].feature_dict, -1)

        return






















