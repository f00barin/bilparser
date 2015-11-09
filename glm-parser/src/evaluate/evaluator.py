from __future__ import division
import logging

logging.basicConfig(filename='glm_parser.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')


class Evaluator():
    def __init__(self):
        self.unlabeled_correct_num = 0
        self.labeled_correct_num = 0
        self.unlabeled_gold_set_size = 0
        self.labeled_gold_set_size = 0
        self.sentno = 0
        return

    def reset(self):
        self.unlabeled_correct_num = 0
        self.labeled_correct_num = 0
        self.unlabeled_gold_set_size = 0
        self.labeled_gold_set_size = 0
        return

    def get_statistics(self):
        return self.unlabeled_correct_num, self.unlabeled_gold_set_size

    def _sent_unlabeled_accuracy(self, result_edge_set, gold_edge_set):
        if isinstance(result_edge_set, list):
            result_edge_set = set(result_edge_set)

        if isinstance(gold_edge_set, list):
            gold_edge_set = set(gold_edge_set)

        intersect_set = result_edge_set.intersection(gold_edge_set)
        correct_num = len(intersect_set)
        gold_set_size = len(gold_edge_set)

#        logging.debug("result edge set: ")
#        logging.debug(result_edge_set)
#        logging.debug("gold edge set: ")
#        logging.debug(gold_edge_set)
#        logging.debug("##############")

        return correct_num, gold_set_size

    def _sent_labeled_accuracy(self, result_edge_set, gold_edge_set):
        if isinstance(result_edge_set, list):
            result_edge_set = set(result_edge_set)

        if isinstance(gold_edge_set, list):
            gold_edge_set = set(gold_edge_set)

        intersect_set = result_edge_set.intersection(gold_edge_set)
        correct_num = len(intersect_set)
        gold_set_size = len(gold_edge_set)

#        logging.debug("result labelled edge set: ")
#        logging.debug(result_edge_set)
#        logging.debug("gold labelled edge set: ")
#        logging.debug(gold_edge_set)
#        logging.debug("##############")

        return correct_num, gold_set_size

    def evaluate(self, data_pool, parser, w_vector, training_time, sfeats,
                 sbfeats):


        logging.debug("Start evaluating ...")
        while data_pool.has_next_data():
            sent = data_pool.get_next_data()
            self.sentno += 1
#            logging.debug("data instance: ")
#            logging.debug(sent.get_word_list())
#            logging.debug(sent.get_edge_list())

            gold_edge_set = \
                set([(head_index,dep_index) for head_index,dep_index,_ in sent.get_edge_list()])

#            sent_len = len(sent.get_word_list())
            test_edge_set = \
               parser.parse(sent, w_vector.get_best_label_score, self.sentno,
                            sfeats, sbfeats)
            #print sent.get_edge_list()

            self.unlabeled_accuracy(test_edge_set, gold_edge_set, True)
            self.labeled_accuracy(test_edge_set, sent.get_edge_list(), True)
        if training_time is not None:
            logging.info("Training time usage(seconds): %f" % (training_time,))
#        logging.info("Feature count: %d" % len(w_vector.data_dict.keys()))
        logging.info("Unlabeled accuracy: %.12f (%d, %d)" % (self.get_acc_unlabeled_accuracy(), self.unlabeled_correct_num, self.unlabeled_gold_set_size))
        logging.info("labeled accuracy: %.12f (%d, %d)" % (self.get_acc_labeled_accuracy(), self.labeled_correct_num, self.labeled_gold_set_size))

        self.unlabeled_attachment_accuracy(data_pool.get_sent_num())
        logging.info("Unlabeled attachment accuracy: %.12f (%d, %d)" % (self.get_acc_unlabeled_accuracy(), self.unlabeled_correct_num, self.unlabeled_gold_set_size))


#   def printParse(self, data_pool, parser, w_vector, training_time):
#        logging.debug("Start evaluating ...")
#        while data_pool.has_next_data():
#            sent = data_pool.get_next_data()
#
#            logging.debug("data instance: ")
#            logging.debug(sent.get_word_list())
#            logging.debug(sent.get_edge_list())
#
#            gold_edge_set = \
#                set([(head_index,dep_index) for head_index,dep_index,_ in sent.get_edge_list()])
#
#            sent_len = len(sent.get_word_list())
#            test_edge_set = \
#               parser.parse(sent, w_vector.get_vector_score)
#
#            print test_edge_set
##            self.unlabeled_accuracy(test_edge_set, gold_edge_set, True)
#

    def unlabeled_accuracy(self, result_edge_set, gold_edge_set, accumulate=False):
        """
        calculate unlabeled accuracy of the glm-parser
        unlabeled accuracy = # of corrected edges in result / # of all corrected edges

        :param result_edge_set: the edge set that needs to be evaluated
        :type result_edge_set: list/set

        :param gold_edge_set: the gold edge set used for evaluating
        :type gold_edge_set: list/set

        :param accumulate:  True -- if evaluation result needs to be remembered
                            False (default) -- if result does not needs to be remembered
        :type accumulate: boolean

        :return: the unlabeled accuracy
        :rtype: float
        """
#        print result_edge_set
        res_edge = [(l[0], l[1]) for l in result_edge_set]
#        print 'result', res_edge
#        print gold_edge_set
        correct_num, gold_set_size =\
            self._sent_unlabeled_accuracy(res_edge, gold_edge_set)
#            self._sent_unlabeled_accuracy(result_edge_set, gold_edge_set)

        if accumulate == True:
            self.unlabeled_correct_num += correct_num
            #correct_num = self.unlabeled_correct_num

            self.unlabeled_gold_set_size += gold_set_size
            #gold_set_size = self.unlabeled_gold_set_size
#            logging.debug("Correct_num: %d, Gold set size: %d, Unlabeled correct: %d, Unlabeled gold set size: %d" % (correct_num, gold_set_size, self.unlabeled_correct_num, self.unlabeled_gold_set_size))

        # WARNING: this function returns a value but the caller does not use it!
        return correct_num / gold_set_size

    def get_acc_unlabeled_accuracy(self):
        return self.unlabeled_correct_num / self.unlabeled_gold_set_size


    def get_acc_labeled_accuracy(self):
        return self.labeled_correct_num / self.labeled_gold_set_size


    def unlabeled_attachment_accuracy(self, sent_num):
        """
        calculate unlabled attachment accuracy of glm-parser
        unlabeled attachment accuracy = # of corrected tokens in result / # of all corrected tokens
        # of corrected tokens in result = # of corrected edges in result + # of sentences in data set
        # of all corrected tokens = # of all corrected edges + # of sentences in data set
        """

        self.unlabeled_correct_num += sent_num
        self.unlabeled_gold_set_size += sent_num

        return


    def labeled_accuracy(self, result_edge_set, gold_edge_set, accumulate=False):
        """
        calculate labeled accuracy of the glm-parser
        labeled accuracy = # of corrected edges in result / # of all corrected edges

        :param result_edge_set: the edge set that needs to be evaluated
        :type result_edge_set: list/set

        :param gold_edge_set: the gold edge set used for evaluating
        :type gold_edge_set: list/set

        :param accumulate:  True -- if evaluation result needs to be remembered
                            False (default) -- if result does not needs to be remembered
        :type accumulate: boolean

        :return: the unlabeled accuracy
        :rtype: float
        """
#        print result_edge_set
#         res_edge = [(l[0], l[1]) for l in result_edge_set]
#        print 'result', res_edge
#        print gold_edge_set
        correct_num, gold_set_size =\
            self._sent_labeled_accuracy(result_edge_set, gold_edge_set)
#            self._sent_unlabeled_accuracy(result_edge_set, gold_edge_set)

        if accumulate == True:
            self.labeled_correct_num += correct_num
            #correct_num = self.unlabeled_correct_num

            self.labeled_gold_set_size += gold_set_size
            #gold_set_size = self.unlabeled_gold_set_size
#            logging.debug("Correct_num: %d, Gold set size: %d, labeled correct: %d, labeled gold set size: %d" % (correct_num, gold_set_size, self.labeled_correct_num, self.labeled_gold_set_size))

        # WARNING: this function returns a value but the caller does not use it!
        return correct_num / gold_set_size


