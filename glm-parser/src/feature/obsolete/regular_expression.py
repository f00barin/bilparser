
import copy
import string

class IndexedStr():
    """
    An auxiliary class for regular expression and similar facilities. Basically
    this is a string with a next_index, which is used by some state machine.
    """
    def __init__(self,s,start_index=0):
        self.s = s
        self.start_index = start_index
        # Used to trace back when parsing fails
        self.stack = []
        return

    def push_index(self):
        """
        Save the current position
        """
        self.stack.append(self.start_index)
        return
    
    def push(self,num):
        """
        Save a mark into the stack
        """
        self.stack.append(num)
        return

    def pop_index(self):
        """
        Pop out a value from the stack to start_index
        """
        self.start_index = self.stack.pop()
        return

    def peak_index(self):
        """
        Pop out a value but do not save it to start_index
        """
        return self.stack.pop()

    def pop_until(self,num):
        """
        Pop the stack until we have seen some value. Then set start_index
        as current stack top
        """
        # Keep popping until we have seen that value
        while self.stack.pop() != num:
            pass
        # Set the current stack top as start_index
        self.start_index = self.stack[-1]

    def proceed(self,s2):
        """
        If s2 is a prefix of the current substring
        (i.e. self[self.start_index:self.start_index + len(s2)]) then return
        True and add start_index by len(s2). If not (either not match or index
        out of bound) then return False

        :return: True if matched, False if not
        :rtype: bool
        """
        length = len(s2)
        if self.try_proceed(s2) == False:
            return False
        else:
            #print "Proceed: ",length
            self.start_index += length
            return True
        
    def try_proceed(self,s2):
        length = len(s2)
        if self.start_index + length > len(self.s):
            return False
        elif self.s[self.start_index:self.start_index + length] != s2:
            return False
        else:
            return True

    def is_end(self):
        if self.start_index >= len(self.s):
            return True
        else:
            return False

    def rewind(self,length):
        self.start_index -= length
        if self.start_index < 0:
            raise ValueError("The index have crossed zero bound")

###############################################################################
########################### The Devil Split Bar ###############################
###############################################################################

class RegExp():
    """
    A simple regular expression recognizer. Do not always recognoize the longest
    possible, but still powerful enough to deal with most of the cases.
    """
    # Three basic node types
    union_node = 0
    concat_node = 1
    star_node = 2
    plus_node = 3
    question_node = 4
    
    def __str__(self):
        return "RegExp Type %d instance" % (self.node_type)

    def __repr__(self):
        return self.__str__()
    
    def __init__(self,initializer=[],node_type=0):
        """
        Make an instance of a regular expression. You can choose to initialize
        it with either a list of words (which will be tested in the order they
        appear in the list, and return the first match), or a string, which
        will be treated as a single token. Essentially a list of words is the
        same as the union operation.

        :param initializer: Either a single string or a list of string or an
                            instance of this class. On the last two cases the
                            list will be shallow copied
        :type initializer: str/list(str)/RegExp
        """
        self.node_type = node_type

        self.parse_method_list = [self.parse_union,self.parse_concat,
                             self.parse_star,self.parse_plus,
                                  self.parse_question]
        
        self.bind_parse_method(node_type)
        
        # single string will be treated as its token
        if isinstance(initializer,str):
            self.token_list = [initializer]
        # List will be copied (shallow copy) to prevent hard-to-debug error
        elif isinstance(initializer,list):
            self.token_list = copy.copy(initializer)
        elif isinstance(initializer,RegExp):
            self.token_list = copy.copy(initializer.token_list)
        else:
            raise TypeError("Do not support other kind of initializers!")
        return

    def bind_parse_method(self,node_type):
        """
        When initializing or changing a node_type, please call this function
        to re-bind the method used to parsing a particular type of node

        Also it will change the node_type accordingly to maintain consistency
        """
        self.parse = self.parse_method_list[node_type]
        self.node_type = node_type
        return

    def append(self,new_token):
        """
        Append a new node to the end of the token list.
        """
        if isinstance(new_token,str) or isinstance(new_token,RegExp):
            # Append it to the last of the token_list
            self.token_list.append(new_token)
        else:
            raise TypeError("Only string or RegExp instance coule be appended")
        return

    def __or__(self,another_node):
        """
        Operator overloading for logical "or" ( a | b ). This will result
        in a new instance of RegExp, which is the union of the two.

        Please notice that the order do have effect on the result. We will
        try to match the first operand first, and if it got matched the
        second will not be processed, even if it is longer (and theoritically
        more optimized one)
        """
        # Copy the first operand (itself) by a copy constructor call
        new_node = RegExp(node_type=RegExp.union_node)
        # Since we will copy the list, the old node is not affected
        new_node.append(self)
        new_node.append(another_node)
        return new_node

    def get_concat(self,another_node):
        """
        Retuen a concatenation node (type = 1), which recognizes the
        concatenation of all nodes in token_list. This will not change this node
        """
        new_node = RegExp(node_type=RegExp.concat_node)
        new_node.append(self)
        new_node.append(another_node)
        return new_node

    def __add__(self,another_node):
        """
        Operator overloading for regexp concatenation.
        """
        return self.get_concat(another_node)

    def get_star(self):
        """
        Return a new node whose type is 3 (RegExp.star_node). Essentially this
        node allows for multiple entries in token_list, and they will be treated
        as concatenation.
        """
        new_node = RegExp([self],RegExp.star_node)
        return new_node

    def star(self):
        return self.get_star()

    def get_plus(self):
        new_node = RegExp([self],RegExp.plus_node)
        return new_node

    def plus(self):
        return self.get_plus()

    def get_question(self):
        new_node = RegExp([self],RegExp.question_node)
        return new_node

    def question(self):
        return self.get_question()

    def parse_union(self,s):
        """
        Parse the union node and return the result if there is (one or more)
        match. The returned string is the longest possible.
        Return None if there is not a match.

        If parse fails s will remain at the same position

        :return: A matched string token, or None if none is matched
        :rtype: str/None
        """
        # Save current index
        s.push_index()

        parse_result = ""
        find_parse = False
        
        for i in self.token_list:
            if isinstance(i,str):
                # Remember that proceed() will increase the index automatically
                if s.try_proceed(i) == True and len(i) > len(parse_result):
                    #print i + ' overrides ' + parse_result
                    parse_result = str(i)
                    find_parse = True
            elif isinstance(i,RegExp):
                s.push_index()
                ret = i.parse(s)
                s.pop_index()
                if ret != None and len(ret) > len(parse_result):
                    parse_result = ret
                    find_parse = True
            else:
                raise TypeError("""No other types other than str and RegExp could
                                be parsed!""")
        # If none of them matches, return None
        if find_parse == False:
            s.pop_index()
            return None
        else:
            s.proceed(parse_result)
            #print "Union result: ",self.token_list,parse_result
            s.peak_index()
            return parse_result

    def parse_concat(self,s):
        """
        Same as parse_union() except that it recognizes a concatenation of nodes

        If parse fails s will remain at the start position before calling this
        """
        s.push_index()
        parse_result = ""
        for i in self.token_list:
            if isinstance(i,str):
                if s.proceed(i) == True:  
                    parse_result += i
                else:   # Any mismatch will cause a None to be returned
                    s.pop_index()
                    return None
            elif isinstance(i,RegExp):
                ret = i.parse(s)
                #print "Concat: ",ret
                if ret == None:
                    s.pop_index()
                    return None
                else:
                    parse_result += ret
            else:
                raise TypeError("""No other types other than str and RegExp could
                                be parsed!""")
        # If none of them matches, return None
        s.peak_index()
        return parse_result

    def parse_star(self,s):
        """
        Same as parse_union() except that it recognizes a star of a string
        (i.e. repeat 0 or more times)
        """
        self.bind_parse_method(RegExp.concat_node)
        parse_result = ""
        # It should not loop forever, since the string must be limited length
        # and self.parse(s) will return None
        while True:
            #print s.start_index
            ret = self.parse(s)
            #print s.start_index
            #print "Star: ",ret
            # We could tolerate an empty string
            if ret == None:
                break
            else:
                parse_result += ret
                
        self.bind_parse_method(RegExp.star_node)
        return parse_result

    def parse_plus(self,s):
        """
        Same as parse_union() except that it regconizes an plus of a string
        (i.e. repeat for 1 or more times)

        If parse fails s will remain at the position before calling this
        """
        s.push_index()
        self.bind_parse_method(RegExp.concat_node)
        parse_result = ""
        ret = self.parse(s)
        # The first pass must be a valid string
        if ret == None:
            s.pop_index()
            # Never forget to add this line to make it restore to its original
            # type of node before exiting!!
            self.bind_parse_method(RegExp.plus_node)
            return None
        else:
            parse_result += ret
        self.bind_parse_method(RegExp.star_node)
        # It must return some string, so we add it directly without any test
        parse_result += self.parse(s)
        self.bind_parse_method(RegExp.plus_node)
        # There must be some valid string if we reach here, so return directly
        s.peak_index()
        return parse_result

    def parse_question(self,s):
        """
        Same as parse_union except that it recognizes zero or one occurrance

        If parse fails s will remain at the position before calling this
        """
        s.push_index()
        self.bind_parse_method(RegExp.concat_node)
        ret = self.parse(s)
        self.bind_parse_method(RegExp.question_node)
        if ret == None:
            s.pop_index()
            return ""
        else:
            s.peak_index()
            return ret
        
class RegBuilder():
    digit = RegExp(list("0123456789"))
    digits = digit.plus()
    digits_or_none = digit.star()
    hex_digit = RegExp(list("0123456789ABCDEFabcdef"))
    
    alpha_lower = RegExp(list("abcdefghijklmnopqrstuvwxyz"))
    alpha_higher = RegExp(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    alpha = alpha_lower | alpha_higher
    alnum = alpha | digit
    underline = RegExp("_")
    al_underline = alpha | underline
    alnum_underline = alnum | underline
    
    # C identifier
    c_ident = al_underline + alnum_underline.star()
    # C decimal integer
    c_decimal = RegExp(["+","-"]).question() + digit.plus()
    # C hex integer
    c_hex = RegExp(["0x","0X"]) + hex_digit.plus()
    # C oct integer
    c_oct = RegExp('0') + digits
    # C integer (all these three)
    c_integer = c_decimal | c_hex | c_oct
    # C float
    c_float = (RegExp(["+","-"]).question() + digits_or_none + RegExp('.') +
               digits + (RegExp(['e','E']) + c_decimal).question()
               )
    single_quote = RegExp("'")
    double_quote = RegExp('"')
    # \\ \n \t \v \r \v \' \"
    escape_char = RegExp('\\') + RegExp(list('\\nvtbr\'"'))
    all_char_no_escape = RegExp(list("""0123456789abcdefghijklmnopqrstuvwxyz
                    ABCDEFGHIJKLMNOPQRSTUVWXYZ!#$%&\'()*+,-./:;<=>?@[\\]^_`
                              {|}~ \t\n\r"""))
    all_char = RegExp(list("""0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLM
                    NOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r"""))
    # C string
    c_str = double_quote + all_char_no_escape.star() + double_quote

    space = RegExp(list(" \n\t\r\v\b"))
    spaces = space.star()

"""
reg_dict = {
    'digit': RegBuilder.digit,
    'digits': RegBuilder.digits,
    'digits_or_none': RegBuilder.digits_or_none,
    'hex_digit': RegBuilder.hex_digit,

    'alpha_lower': RegBuilder.alpha_lower,
    'alpha_higher': RegBuilder.alpha_higher,
    'alpha': RegBuilder.alpha,
    'alnum': RegBuilder.alnum,
    'underline': RegBuilder.underline,
    'al_underline': RegBuilder.al_underline,
    'alnum_underline': RegBuilder.alnum_underline,

    'c_ident': RegBuilder.c_ident,
    'c_decimal': RegBuilder.c_decimal,
    'c_hex': RegBuilder.c_hex,
    'c_oct': RegBuilder.c_oct,
    'c_integer': RegBuilder.c_integer,
    'c_float': RegBuilder.c_float,

    'single_quote': RegBuilder.single_quote,
    'double_quote': RegBuilder.double_quote,
    'escape_char': RegBuilder.escape_char,
    'all_char_no_escape': RegBuilder.all_char_no_escape,
    'all_char': RegBuilder.all_char,

    'c_str': RegBuilder.c_str,
    'space': RegBuilder.space,
    'spaces': RegBuilder.spaces,
    }
"""

reg_dict = RegBuilder.__dict__
    
if __name__ == "__main__":
    reg = RegExp(['me','ow '],RegExp.plus_node)
    reg2 = RegExp(["My"," master"],0)
    s = IndexedStr("!!!@")
    reg3 = (reg + reg2) | RegExp(['!']).plus()
    s1 = IndexedStr('12345')
    print RegBuilder.c_decimal.parse(s1)
