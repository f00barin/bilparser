# -*- coding: utf-8 -*-
import sqlite3
import os.path
import logging
from itertools import permutations
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
        for i in range(max_iter):
            logging.debug("Iteration: %d" % i)
            logging.debug("Data size: %d" % len(data_pool.data_list))


            while data_pool.has_next_data():
                self.sentence += 1
                if os.path.isfile('training.db'):
                    db = sqlite3.connect('training.db')
                else:
                    db = sqlite3.connect('training.db')
                    cur = db.cursor()
                    cur.execute('''CREATE TABLE features (sentno integer,  head
                                integer, mod integer, feats text)''')
                    db.commit()

                data_instance = data_pool.get_next_data()
#                print self.sentence
#                for h, m in permutations(range(len(data_instance.word_list)),2):
##                    print h,m
#                    cur = db.cursor()
#                    cur.execute("INSERT INTO features VALUES (?,?,?,?)",
#                               (self.sentence, h, m, '###join###'.join(data_instance.get_local_vector(h,m)[0])))
#                    db.commit()
#
#                for h,m in hmlist:
#                    print 'this is %s, %s' % (h,m), data_instance.get_local_vector(h,m)


                gold_global_vector = data_instance.gold_global_vector
#                print gold_global_vector.keys()
                current_global_vector = f_argmax(data_instance, self.sentence)
#                print current_global_vector.keys()
                self.update_weight(current_global_vector, gold_global_vector)

            data_pool.reset_index()
            if d_filename is not None:
                if i % dump_freq == 0 or i == max_iter - 1:
                    self.w_vector.dump(d_filename + "_Iter_%d.db"%i)


    def update_weight(self, current_global_vector, gold_global_vector):
        # otherwise, the gold_global_vector will change because of the change in weights
        for key in gold_global_vector.keys():
            self.w_vector.data_dict[key].iadd(gold_global_vector[key].feature_dict)

#        print 'Gold ', gold_global_vector
#        print 'Current ', current_global_vector

        for key in current_global_vector.iterkeys():
            self.w_vector.data_dict[key].iaddc(current_global_vector[key].feature_dict, -1)

        return






















