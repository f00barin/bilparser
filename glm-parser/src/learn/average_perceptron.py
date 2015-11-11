from __future__ import division
import logging
import multiprocessing
from hvector._mycollections import mydefaultdict
from hvector.mydouble import mydouble
from weight.weight_vector import *
# Time accounting and control
import debug.debug
from evaluate.evaluator import *
import time
import sys

logging.basicConfig(filename='glm_parser.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

class AveragePerceptronLearner():

    def __init__(self, w_vector, max_iter=1):
        """
        :param w_vector: A global weight vector instance that stores
         the weight value (float)
        :param max_iter: Maximum iterations for training the weight vector
         Could be overridden by parameter max_iter in the method
        :return: None
        """
        logging.debug("Initialize AveragePerceptronLearner ... ")
        self.w_vector = w_vector
        self.max_iter = max_iter
        self.labels = ['AMOD','DEP','NMOD','OBJ','P','PMOD','PRD','ROOT','SBAR','SUB','VC','VMOD']
        self.weight_sum_dict = dict.fromkeys(self.labels, mydefaultdict(mydouble))
        self.last_change_dict = dict.fromkeys(self.labels, mydefaultdict(mydouble))
        self.c = 1
        self.evaluator = Evaluator()
        self.sentence = 0
        return

    def sequential_learn(self, f_argmax, data_pool=None, max_iter=-1, d_filename=None, dump_freq = 1, sfeats=None, sbfeats=None,test_data=[], parser=None, tfeats=None, tbfeats=None):
        if max_iter <= 0:
            max_iter = self.max_iter

        logging.debug("Starting sequential train ... ")

        # sigma_s
        for l in self.labels:
            self.weight_sum_dict[l].clear()
            self.last_change_dict[l].clear()
        self.c = 1

        # for t = 1 ... T
        for t in range(max_iter): 
            logging.debug("Iteration: %d" % t)
            logging.debug("Data size: %d" % len(data_pool.data_list))
            sentence_count = 1
            argmax_time_total = 0.0

            start_time = time.clock()
            # for i = 1 ... m
            while data_pool.has_next_data():
                print("Iteration: %d, Sentence %d" % (t, sentence_count))
                sentence_count += 1
                # Calculate yi' = argmax
                data_instance = data_pool.get_next_data()
                self.sentence += 1 
                print self.sentence
                gold_global_vector = data_instance.gold_global_vector

                if debug.debug.time_accounting_flag is True:
                    before_time = time.clock()
                    current_global_vector = f_argmax(data_instance, self.sentence)
                    after_time = time.clock()
                    time_usage = after_time - before_time
                    argmax_time_total += time_usage
                    print("Sentence length: %d" % (len(data_instance.word_list) - 1))
                    print("Time usage: %f" % (time_usage, ))
                    logging.debug("Time usage %f" % (time_usage, ))
                else:
                    # Just run the procedure without any interference
                    current_global_vector = f_argmax(data_instance, self.sentence)
                delta_global_vector = {}
                for l in self.labels:
                    if l in gold_global_vector and l in current_global_vector:
                        delta_global_vector[l] = gold_global_vector[l] - current_global_vector[l]
                    elif l in gold_global_vector and l not in current_global_vector: 
                        delta_global_vector[l] = gold_global_vector[l]
                    elif l not in gold_global_vector and l in current_global_vector:
                        delta_global_vector[l] =  current_global_vector[l] - current_global_vector[l] - current_global_vector[l]

                
                # update every iteration (more convenient for dump)
                if data_pool.has_next_data():
                    # i yi' != yi
                    for l in delta_global_vector.keys():
                        if l not in current_global_vector or l not in gold_global_vector or not current_global_vector[l] == gold_global_vector[l]:
                            # for each element s in delta_global_vector
                            for s in delta_global_vector[l].keys():
                                self.weight_sum_dict[l][s] += self.w_vector[l][s] * (self.c - self.last_change_dict[l][s])
                                self.last_change_dict[l][s] = self.c
                            
                            # update weight and weight sum
                            self.w_vector.data_dict[l].iadd(delta_global_vector[l].feature_dict)
                            self.weight_sum_dict[l].iadd(delta_global_vector[l].feature_dict)


                else:
                    for l in delta_global_vector.keys():
                        for s in self.last_change_dict[l].keys():
                            self.weight_sum_dict[l][s] += self.w_vector[l][s] * (self.c - self.last_change_dict[l][s])
                            self.last_change_dict[l][s] = self.c
                            
                        if l not in current_global_vector or l not in gold_global_vector or not current_global_vector[l] == gold_global_vector[l]:
                            self.w_vector.data_dict[l].iadd(delta_global_vector[l].feature_dict)
                            self.weight_sum_dict[l].iadd(delta_global_vector[l].feature_dict)

                self.c += 1

                if debug.debug.log_feature_request_flag is True:
                    data_instance.dump_feature_request("%s" % (sentence_count, ))

                # If exceeds the value set in debug config file, just stop and exit
                # immediatel y
                if sentence_count > debug.debug.run_first_num > 0:
                    print("Average time for each sentence: %f" % (argmax_time_total / debug.debug.run_first_num))
                    logging.debug("Average time for each sentence: %f" % (argmax_time_total / debug.debug.run_first_num))
                    data_pool.reset_index()
                    argmax_time_total = 0.0

            # End while(data_pool.has_next_data())

            # Reset index, while keeping the content intact
            data_pool.reset_index()
            if not test_data == []:
                end_time = time.clock()
                ttime = end_time - start_time
                self.evaluator.evaluate(test_data, parser, self.w_vector, ttime,tfeats, tbfeats) 
            sentence_count = 1


            if d_filename is not None:
                if t % dump_freq == 0 or t == max_iter - 1:
                    p_fork = multiprocessing.Process(
                        target=self.dump_vector,
                        args=(d_filename, t))
                
                    p_fork.start()
                    #self.w_vector.dump(d_filename + "_Iter_%d.db"%t)
        for l in self.labels: 
            self.w_vector.data_dict[l].clear()

        self.avg_weight(self.w_vector, self.c - 1)

        return

    def avg_weight(self, w_vector, count):
        for l in self.labels:
            if count > 0:
                w_vector.data_dict[l].iaddc(self.weight_sum_dict[l], 1 / count)
            
    def dump_vector(self, d_filename, i):
        d_vector = WeightVector()
        self.avg_weight(d_vector, self.c-1)
        d_vector.dump(d_filename + "_Iter_%d.db"%i)
        d_vector.data_dict.clear()
