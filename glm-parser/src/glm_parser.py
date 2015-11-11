# -*- coding: utf-8 -*-

#
# Global Linear Model Parser
# Simon Fraser University
# NLP Lab
#
# Author: Yulan Huang, Ziqi Wang, Anoop Sarkar
# (Please add on your name if you have authored this file)
#

from data.data_pool import *
import shelve
import cPickle
from evaluate.evaluator import *

from weight.weight_vector import *

import debug.debug
import debug.interact

import time

class GlmParser():
    def __init__(self, train_section=[], test_section=[], data_path="../../penn-wsj-deps/",
                 l_filename=None, max_iter=1,
                 learner=None,
                 fgen=None,
                 parser=None):
        self.max_iter = max_iter
        self.data_path = data_path
        self.w_vector = WeightVector(l_filename)

#        self.trainfeats = shelve.open('/home/usuaris/pranava/bilparser/glm-parser/src/TRAINING.db', flag='r')
#        self.btfeats = cPickle.load(open('/home/usuaris/pranava/bilparser/glm-parser/src/biltrainfeats.pkl', 'rb'))
#        self.sfeats = shelve.open('/home/usuaris/pranava/bilparser/glm-parser/src/VALIDATION.db', flag='r')
#        self.sbfeats = cPickle.load(open('/home/usuaris/pranava/bilparser/glm-parser/src/vbfeatures.pkl', 'rb'))

#        self.btfeats = cPickle.load(open('/home/usuaris/pranava/bilparser/glm-parser/src/trainingl1.pkl', 'rb'))
#        self.sbfeats = cPickle.load(open('/home/usuaris/pranava/bilparser/glm-parser/src/validatl1.pkl', 'rb'))
        self.btfeats = {}
        self.trainfeats = {}
        self.sfeats = {}
        self.sbfeats = {}


        if fgen is not None:
            # Do not instanciate this; used elsewhere
            self.fgen = fgen
        else:
            raise ValueError("You need to specify a feature generator")

        self.train_data_pool = DataPool(train_section, data_path, fgen=self.fgen)
        self.test_data_pool = DataPool(test_section, data_path, fgen=self.fgen)

        self.parser = parser()

        if learner is not None:
            self.learner = learner(self.w_vector, max_iter)
            #self.learner = PerceptronLearner(self.w_vector, max_iter)
        else:
            raise ValueError("You need to specify a learner")

        self.evaluator = Evaluator()

    def evaluate(self, training_time,  test_section=[]):
        if not test_section == []:
            test_data_pool = DataPool(test_section, self.data_path, fgen=self.fgen)
        else:
            test_data_pool = self.test_data_pool
        self.evaluator.evaluate(test_data_pool, self.parser, self.w_vector,
                                training_time, self.sfeats, self.sbfeats)


    def sequential_train(self, train_section=[], max_iter=-1, d_filename=None,
                         dump_freq = 1, test_section=[]):
        if not train_section == []:
            train_data_pool = DataPool(train_section, self.data_path, fgen=self.fgen)
        else:
            train_data_pool = self.train_data_pool

        if max_iter == -1:
            max_iter = self.max_iter
        if not test_section == []:
            test_data_pool = DataPool(test_section, self.data_path, fgen=self.fgen)
        self.learner.sequential_learn(self.compute_argmax, train_data_pool,
                                      max_iter, d_filename, dump_freq,
                                      self.trainfeats, self.btfeats, test_data_pool, self.parser, 
                                      self.sfeats, self.sbfeats)

    def printParses(self, training_time,  test_section=[]):
        if not test_section == []:
            test_data_pool = DataPool(test_section, self.data_path, fgen=self.fgen)
        else:
            test_data_pool = self.test_data_pool

        self.evaluator.printParse(test_data_pool, self.parser, self.w_vector, training_time)




    def compute_argmax(self, sentence, sentno):
        current_edge_set = self.parser.parse(sentence,
                                             self.w_vector.get_best_label_score,
                                             sentno, self.trainfeats,
                                             self.btfeats)

#        print current_edge_set
        sentence.set_current_global_vector(current_edge_set, self.btfeats)
        current_global_vector = sentence.current_global_vector

        return current_global_vector

HELP_MSG =\
"""

options:
    -h:     Print this help message

    -b:     Begin section for training
            (You need to specify both begin and end section)

    -e:     End section for training
            (You need to specify both begin and end section)

    -t:     Test sections for evaluation
            Use section id to specify them, separated with comma (,)
            i.e.  "-t 1,2,3,4,55"
            If not specified then no evaluation will be conducted

    -p:     Path to data files (to the parent directory for all sections)
            default "./penn-wsj-deps/"

    -l:     Path to an existing weight vector dump file
            example: "./Weight.db"

    -d:     Path for dumping weight vector. Please also specify a prefix
            of file names, which will be added with iteration count and
            ".db" suffix when dumping the file

            example: "./weight_dump", and the resulting files could be:
            "./weight_dump_Iter_1.db",
            "./weight_dump_Iter_2.db"...

    -f:     Frequency of dumping weight vector. The weight vecotr of last
            iteration will always be dumped
            example: "-i 6 -f 2"
            weight vector will be dumpled at iteration 0, 2, 4, 5.

    -i:     Number of iterations
            default 1

    -a:     Turn on time accounting (output time usage for each sentence)
            If combined with --debug-run-number then before termination it also
            prints out average time usage

    --learner=
            Specify a learner for weight vector training

            default "average_perceptron"; alternative "perceptron"

    --fgen=
            Specify feature generation facility by a python file name (mandatory).
            The file will be searched under /feature directory, and the class
            object that has a get_local_vector() interface will be recognized
            as the feature generator and put into use automatically.

            If multiple eligible objects exist, an error will be reported.

            For developers: please make sure there is only one such class object
            under fgen source files. Even import statement may introduce other
            modules that is eligible to be an fgen. Be careful.

            default "english_2nd_fgen"; alternative "english_1st_fgen"

    --parser=
            Specify the parser using parser module name (i.e. .py file name without suffix).
            The recognition rule is the same as --fgen switch. A valid parser object
            must possess "parse" attribute in order to be recognized as a parser
            implementation.

            Some parser might not work correctly with the infrastructure, which keeps
            changing all the time. If this happens please file an issue on github page

            default "ceisner3"; alternative "ceisner"

    --debug-run-number=[int]
            Only run the first [int] sentences. Usually combined with option -a to gather
            time usage information
            If combined with -a, then time usage information is available, and it will print
            out average time usage after each iteration
            *** Caution: Overrides -t (no evaluation will be conducted), and partially
            overrides -b -e (Only run specified number of sentences) -i (Run forever)

    --interactive
            Use interactive version of glm-parser, in which you have access to some
            critical points ("breakpoints") in the procedure

    --log-feature-request
            Log each feature request based on feature type. This is helpful for
            analyzing feature usage and building feature caching utility.
            Upon exiting the main program will dump feature request information
            into a file "feature_request.log"

"""

def get_class_from_module(attr_name, module_path, module_name,
                          silent=False):
    """
    This procedure is used by argument processing. It returns a class object
    given a module path and attribute name.

    Basically what it does is to find the module using module path, and then
    iterate through all objects inside that module's namespace (including those
    introduced by import statements). For each object in the name space, this
    function would then check existence of attr_name, i.e. the desired interface
    name (could be any attribute fetch-able by getattr() built-in). If and only if
    one such desired interface exists then the object containing that interface
    would be returned, which is mostly a class, but could theoretically be anything
    object instance with a specified attribute.

    If import error happens (i.e. module could not be found/imported), or there is/are
    no/more than one interface(s) existing, then exception will be raised.

    :param attr_name: The name of the desired interface
    :type attr_name: str
    :param module_path: The path of the module. Relative to src/
    :type module_path: str
    :param module_name: Name of the module e.g. file name
    :type module_name: str
    :param silent: Whether to report existences of interface objects
    :return: class object
    """
    # Load module by import statement (mimic using __import__)
    # We first load the module (i.e. directory) and then fetch its attribute (i.e. py file)
    module_obj = getattr(__import__(module_path + '.' + module_name,
                                    globals(),   # Current global
                                    locals(),    # Current local
                                    [],          # I don't know what is this
                                    -1), module_name)
    intf_class_count = 0
    intf_class_name = None
    for obj_name in dir(module_obj):
        if hasattr(getattr(module_obj, obj_name), attr_name):
            intf_class_count += 1
            intf_class_name = obj_name
            if silent is False:
                print("Interface object %s detected\n\t with interface %s" %
                      (obj_name, attr_name))
    if intf_class_count < 1:
        raise ImportError("Interface object with attribute %s not found" % (attr_name, ))
    elif intf_class_count > 1:
        raise ImportError("Could not resolve arbitrary on attribute %s" % (attr_name, ))
    else:
        class_obj = getattr(module_obj, intf_class_name)
        return class_obj


MAJOR_VERSION = 1
MINOR_VERSION = 0

if __name__ == "__main__":
    import getopt, sys

    train_begin = -1
    train_end = -1
    testsection = []
    max_iter = 1
    test_data_path = "../../penn-wsj-deps/"  #"./penn-wsj-deps/"
    l_filename = None
    d_filename = None
    dump_freq = 1

    # Default learner
    #learner = AveragePerceptronLearner
    learner = get_class_from_module('sequential_learn', 'learn', 'average_perceptron',
                                    silent=True)
    # Default feature generator (2dnd, english)
    fgen = get_class_from_module('get_local_vector', 'feature', 'english_2nd_fgen',
                                 silent=True)
    parser = get_class_from_module('parse', 'parse', 'ceisner3',
                                   silent=True)
    # parser = parse.ceisner3.EisnerParser
    # Main driver is glm_parser instance defined in this file
    glm_parser = GlmParser

    try:
        opt_spec = "ahb:e:t:i:p:l:d:f:"
        long_opt_spec = ['fgen=', 'learner=', 'parser=', 'debug-run-number=',
                         'force-feature-order=', 'interactive',
                         'log-feature-request']
        opts, args = getopt.getopt(sys.argv[1:], opt_spec, long_opt_spec)

        print '=========================='
        for opt, value in opts:
            if opt == "-h":
                print("")
                print("Global Linear Model (GLM) Parser")
                print("Version %d.%d" % (MAJOR_VERSION, MINOR_VERSION))
                print(HELP_MSG)
                sys.exit(0)
            elif opt == "-b":
                train_begin = int(value)
            elif opt == "-e":
                train_end = int(value)
            elif opt == "-t":
                testsection = [int(sec) for sec in value.split(',')]
            elif opt == "-p":
                test_data_path = value
            elif opt == "-i":
                max_iter = int(value)
            elif opt == "-l":
                l_filename = value
            elif opt == "-d":
                d_filename = value
            elif opt == '-a':
                print("Time accounting is ON")
                debug.debug.time_accounting_flag = True
            elif opt == '-f':
                dump_freq = int(value)
            elif opt == '--debug-run-number':
                debug.debug.run_first_num = int(value)
                if debug.debug.run_first_num <= 0:
                    raise ValueError("Illegal integer: %d" %
                                     (debug.debug.run_first_num, ))
                else:
                    print("Debug run number = %d" % (debug.debug.run_first_num, ))
            elif opt == "--learner":
                learner = get_class_from_module('sequential_learn', 'learn', value)
                print("Using learner: %s (%s)" % (learner.__name__, value))
            elif opt == "--fgen":
                fgen = get_class_from_module('get_local_vector', 'feature', value)
                print("Using feature generator: %s (%s)" % (fgen.__name__, value))
            elif opt == "--parser":
                parser = get_class_from_module('parse', 'parse', value)
                print("Using parser: %s (%s)" % (parser.__name__, value))
            elif opt == '--interactive':
                glm_parser.sequential_train = debug.interact.glm_parser_sequential_train_wrapper
                glm_parser.evaluate = debug.interact.glm_parser_evaluate_wrapper
                glm_parser.compute_argmax = debug.interact.glm_parser_compute_argmax_wrapper
                DataPool.get_data_list = debug.interact.data_pool_get_data_list_wrapper
                learner.sequential_learn = debug.interact.average_perceptron_learner_sequential_learn_wrapper
                print("Enable interactive mode")
            elif opt == '--log-feature-request':
                debug.debug.log_feature_request_flag = True
                print("Enable feature request log")
            else:
                print "Invalid argument, try -h"
                sys.exit(0)

        gp = glm_parser(data_path=test_data_path, l_filename=l_filename,
                        learner=learner,
                        fgen=fgen,
                        parser=parser)

        training_time = None
        if train_end >= train_begin >= 0:
            start_time = time.clock()
            gp.sequential_train([(train_begin, train_end)], max_iter, d_filename, dump_freq, testsection)
            end_time = time.clock()
            training_time = end_time - start_time

#        if not testsection == []:
#            gp.evaluate(training_time, testsection)
#
    except getopt.GetoptError, e:
        print("Invalid argument. \n")
        print(HELP_MSG)
        # Make sure we know what's the error
        raise

