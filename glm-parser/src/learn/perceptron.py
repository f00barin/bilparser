# -*- coding: utf-8 -*-

#import os.path
import time
import logging
from evaluate.evaluator import *
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
        self.evaluator = Evaluator()
        return

    def sequential_learn(self, f_argmax, data_pool=None, max_iter=-1,
                         d_filename=None, dump_freq = 1, sfeats=None,
                         sbfeats=None, test_data=[], parser=None, tfeats=None, tbfeats=None):
        if max_iter <= 0:
            max_iter = self.max_iter

        logging.debug("Starting sequantial train...")


        for i in range(max_iter):
            logging.debug("Iteration: %d" % i)
            logging.debug("Data size: %d" % len(data_pool.data_list))
            self.sentence = 0
            start_time = time.clock()
            while data_pool.has_next_data():
                self.sentence += 1
                print 'iteration: ', i, '; sent no.: ', self.sentence
                data_instance = data_pool.get_next_data()
                gold_global_vector = data_instance.gold_global_vector
                current_global_vector = f_argmax(data_instance, self.sentence)
                self.update_weight(current_global_vector, gold_global_vector)
            data_pool.reset_index()
            if not test_data == []:
                end_time = time.clock()
                ttime = end_time - start_time
                self.evaluator.evaluate(test_data, parser, self.w_vector, ttime,tfeats, tbfeats) 

#                print gold_global_vector['NMOD']
#                if 'OBJ' in current_global_vector:
#                    print current_global_vector['OBJ']


            if d_filename is not None:
                if i % dump_freq == 0 or i == max_iter - 1:
                    self.w_vector.dump(d_filename + "_Iter_%d.db"%i)


    def update_weight(self, current_global_vector, gold_global_vector):
        # otherwise, the gold_global_vector will change because of the change in weights
        for key in gold_global_vector.keys():
            self.w_vector.data_dict[key].iadd(gold_global_vector[key].feature_dict)

        for key in current_global_vector.iterkeys():
            self.w_vector.data_dict[key].iaddc(current_global_vector[key].feature_dict, -1)
        return





#            db = sqlite3.connect('validation_sql.db')
#            cur = db.cursor()
#            cur.execute('''CREATE TABLE features (sentno integer,  head
##                        integer, mod integer, feats text)''')
##            db.commit()
#            db.close()
#                    print h,m
#                    cur = db.cursor()
#                    cur.execute("INSERT INTO features VALUES (?,?,?,?)",
#                               (self.sentence, h, m, '###join###'.join(data_instance.get_local_vector(h,m)[0])))
#                    db.commit()
#
#                for h,m in hmlist:
#                    print 'this is %s, %s' % (h,m), data_instance.get_local_vector(h,m)
##                    cur.execute("INSERT INTO features VALUES (?,?,?,?)",
##                               (self.sentence, h, m,
##                                '###join###'.join(data_instance.get_local_vector(h,m,
##                                                                                 self.sentence)[0])))
##                db.commit()
#





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
#                                        (str(5, 0, headlist[hi], modlist[mi],label), scoremat[hi, mi])
#                                else:
#                                    s[label] = {(headlist[hi], modlist[mi]): \
#                                         (str(5, 0, headlist[hi], modlist[mi],
#                                              \
#                                              label), scoremat[hi, mi])}
#                    s.sync()
#        s.close()
#





#            headlist = [w.strip().split()[0] for w in
#                        open('penn-wsj-deps/reps/07/ptb.train.original.heads.reps')]
#            modlist = [w.strip().split()[0] for w in
#                       open('penn-wsj-deps/reps/07/ptb.train.original.mods.reps')]
##            labels = ['AMOD','DEP','NMOD','OBJ','P','PMOD','PRD','SBAR','SUB','VC','VMOD']
#            labels = ['VC']
#            s = shelve.open('lbfeatures'+labels[0]+'.db', flag='c', writeback=True)
#            for label in labels:
#                hmmat = np.load('penn-wsj-deps/reps/bt.'+label+'.npy')
#                while data_pool.has_next_data():
#                    self.sentence += 1
#                    sentno = str(self.sentence)
#                    data_instance = data_pool.get_next_data()
#                    print label, self.sentence
#                    for h, m in permutations(data_instance.word_list,2):
#                        if h in headlist and m in modlist:
#                            head = headlist.index(h)
#                            mod = modlist.index(m)
#                            if label in s:
#                                if not (h,m) in s[label]:
#                                    s[label][(h,m)] = (str((5, 0, h, m, label)),
#                                                    hmmat[head, mod])
#                            else:
#                                s[sentno] = {(h,m):(str((5, 0, h, m, label)),
#                                                    hmmat[head, mod])}
#                    s.sync()
#                del(hmmat)
#            s.sync()
#            s.close()











