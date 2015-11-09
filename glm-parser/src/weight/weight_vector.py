#
# Global Linear Model Parser
# Simon Fraser University
# NLP Lab
#
# Author: Yulan Huang, Ziqi Wang, Anoop Sarkar
# (Please add on your name if you have authored this file)
#

#import sys
from debug.debug import local_debug_flag
import datetime
if local_debug_flag is False:
    from hvector._mycollections import mydefaultdict
    from hvector.mydouble import mydouble
else:
    print("Local debug is on. Use dict() and float()")
    mydefaultdict = dict
    mydouble = float

import logging

logging.basicConfig(filename='glm_parser.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

class WeightVector():
    """
    A dictitionary-like object. Used to facilitate class FeatureSet to
    store the features.

    Callables inside the class:

    dump()  - Dump the content of the data object into memory. When we are using
              memory dict it will call Pickle to do that. When we are using

    load()  - Load the content of a disk file into the memory. When we are using
              memory dict it will call Pickle to do the load. And when we are
              using shelves it has no effect, since shelves itself is persistent
              object.

    keys()  - Return a list of keys in the dictionary.

    has_key() - Check whether a given key is in the dictionary.

    Please notice that there is no open() method as in other similar classes.
    Users must provide a file name as well as an operating mode to support
    both persistent and non-persistent (or semi-persistent) operations.
    """
    def __init__(self, filename=None):
        """
        :param store_type: Specify the type of database you want to use
        :type store_type: int
        :param filename: The file name of the database file. If you are using
        memory_dict then this could be given here or in dump(). However if
        you are using shelve or other possible extensions, you must provide
        a file name here in order to establish the connection to the database.
        :type filename: str
        """

        # change to hvector
        self.data_dict = {}
#        self.data_dict = mydefaultdict(mydouble)
        self.labels = ['AMOD','DEP','NMOD','OBJ','P','PMOD','PRD','ROOT','SBAR','SUB','VC','VMOD']
#        self.labels = ['AMOD','NMOD','PMOD','SBAR','VMOD']
        for l in self.labels:
            self.data_dict[l] = mydefaultdict(mydouble)
        if not filename == None:
            self.load(filename)

        return

    #def get_sub_vector(self, key_list):
        # TODO figure out a more efficient way
    #    sub_vector = mydefaultdict(mydouble)
    #    for k in key_list:
    #        sub_vector[k] = self.data_dict[k]
    #    return sub_vector

#    def get_vector_score(self, fv, label):
#        score = self.data_dict[label].evaluate(fv)
#        return score

    def get_best_label_score(self, fullvec):
        bestlabel = None
        bestscore = None
        fv = fullvec[0]
        (head, mod) = fullvec[1]
        sbdict = fullvec[4]
        import re
        for l in self.labels:
            score = self.data_dict[l].evaluate(fv)
#            for k in self.data_dict[l].iterkeys():
#                print k, self.data_dict[l][k]
#            if l in ['AMOD','DEP','NMOD','OBJ','P','PMOD','PRD','ROOT','SBAR','SUB','VC','VMOD']:
##            if l in ['OBJ','SUB','VC','VMOD'] :
            if l in sbdict:
                if (head, mod) in sbdict[l]:
                    if str((5, 0, l)) in self.data_dict[l]:
#                        if sbdict[l][(head, mod)] > 0:
                        score += (self.data_dict[l][str((5, 0, l))] * sbdict[l][(head,mod)])
            if bestscore == None or bestscore < score:
                bestlabel = l
                bestscore = score
        return bestscore, bestlabel

    def get_vector_score(self, fullvec):
        bestlabel = None
        bestscore = None
        fv = fullvec[0]
        (head, mod) = fullvec[1]
        sbdict = fullvec[4]

        for l in self.labels:
            score = self.data_dict[l].evaluate(fv)
            if l in ['AMOD','NMOD','PMOD','VMOD']:
                if l in sbdict:
                    if (head, mod) in sbdict[l]:
                            if str((5,0,head, mod, l)) in self.data_dict[l]:
                                score += (self.data_dict[l][str((5,0,l))] * sbdict[l][(head,mod)])
            if bestscore == None or bestscore < score:
                bestlabel = l
                bestscore = score
        return bestscore, bestlabel



    def load(self,filename):
        """
        Load the dumped memory dictionary Pickle file into memory. Essentially
        you can do this with a shelve object, however it does not have effect,
        since shelve file has been opened once you created the instance.

        Parameter is the same as constructor (__init__).
        """
        logging.debug("Loading Weight Vector from %s " % filename)
        fp = open(filename,"r")
        for line in fp:
            label, key, val = line.strip().split('\t')
            if label in self.data_dict:
                self.data_dict[label].update({key: val})
            else:
                print 'something is wrong'
        #print self.data_dict
    #self.data_dict = pickle.load(fp)
        fp.close()
        return

    def __getitem__(self,index):
        return self.data_dict[index]

    def __setitem__(self,index,value):
        self.data_dict[index] = value
        return

    def has_key(self,index):
        return self.data_dict.has_key(index)

    def pop(self,key):
        self.data_dict.pop(key)
        return

    def keys(self):
        """
        Return a list of dictionary keys. This operation is not as expensive
        as the shelve keys() method, so we separate them.
        """
        return self.data_dict.keys()

    def dump(self,filename):
        """
        Called when memory dictionary is used. Dump the content of the dict
        into a disk file using Pickle
        """
        if filename == None:
            print "Skipping dump ..."
            return
        else:
            filename = filename + datetime.datetime.now().strftime("%H%M%f")

        logging.debug("Dumping Weight Vector to %s " % filename)
        logging.debug("Total Feature Num: %d " % len(self.data_dict))
        fp = open(filename,"w")
        for KEY, VAL  in self.data_dict.items():
            for key, value in VAL.items(): 
                fp.write('%s\t%s\t%s\n' % (KEY, key, value))
#                fp.write(str(key) + "    " + str(self.data_dict[key]) + "\n")
        #pickle.dump(self.data_dict,fp,-1)
        fp.close()
        return

##    def get_vector_score(self, fv, label):
#        bestlabel = None
#        bestscore = None
#        fv = fullvec[0]
##        h = fullvec[1][0]
##        m = fullvec[1][1]
##        hm = np.zeros(11)
#        for l in self.labels:
#            score = self.data_dict[l].evaluate(fv)
##            score += self.data_dict[l][str((5,0,h,m,l))] * hm[0]
##            for val in xrange(1,6):
##                score += self.data_dict[l][str((5,1,h,l))] * hm[val]
##            for val in xrange(6,11):
##                score += self.data_dict[l][str((5,2,m,l))] * hm[val]
##            #print self.data_dict[l][str((5,0,h,m))]
#            if bestscore == None or bestscore < score:
#                bestlabel = l
#                bestscore = score
#        return bestscore, bestlabel
#
#
