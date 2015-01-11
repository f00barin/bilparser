#!/usr/bin/env python

import sys
import os
from collections import defaultdict
import numpy as np

SHIFT = 0
RIGHT = 1
LEFT = 2
START = ['-START-', '-START2-']
END = ['-END-', '-END2-']

class DefaultList(list):
    def __init__(self, default=None):
        self.default = default
        list.__init__(self)

    def __getitem__(self, index):
        try:
            return list.__getitem__(self, index)
        except IndexError:
            return self.default

class Parse(object):

    def __init__(self, n):

        self.n = n
        self.heads = [None] * (n - 1)
        self.labels = [None] * (n - 1)
        self.lefts = []
        self.rights = []
        for i in xrange(n+1):
            self.lefts.append(DefaultList(0))
            self.rights.append(DefaultList(0))

    def add(self, head, child, label=None):

        self.heads[child] = head
        self.labels[child] = label

        if child < head:
            self.lefts[head].append(child)
        else:
            self.rights[head].append(child)


class Parser(object):

    def __init__(self, load=True):

        model_dir = os.path.dirname(__file__)
        self.model = Perceptron(MOVES)
        if load:
            self.model.load(os.path.join(model_dir, 'parser.pickle'))
        self.tagger = PerceptronTagger(load=load)
        self.confusion_matrix = defaultdict(lambda: defaultdict(int))

    def save(self):
        self.model.save(os.path.join(os.path.dirname(__file__), 'parser.pickle'))
        self.tagger.save()

    def parse(self, words):
        n = len(words)
        i = 2
        stack = [1]
        parse = Parse(n)
        tags = self.tagger.tag(words)

        while stack or (i+1) < n:
            features = extract_features(words, tags, i, n, stack, parse)
            scores = self.model.score(features)
            valid_moves = get_valid_moves(i, n, len(stack))
            guess = max(valid_moves, key=lambda move: scores[move])
            i = transition(guess, i, stack, parse)
        return tags, parse.heads

    def train_one(self, itn, words, gold_tags, gold_heads):
        n = len(words)
        i = 2
        stack = [1]
        parse = Parse(n)
        while stack or (i + 1) < n:
            features = extract_features(words, tags, i, n, stack, parse)
            scores = self.model.score(features)
            valid_moves = get_valid_moves(i, n , len(stack))
            gold_moves = get_gold_moves(i, n, stack, parse.heads, gold_heads)
            guess = max(valid_moves, key=lambda move: scores[move])
            assert gold_moves
            best = max(gold_moves, key=lambda move: scores[move])
            self.model.update(best, guess, features)
            i = transition(guess, i, stack, parse)
            self.confusion_matrix[best][guess] += 1

        return len([i for i in range(n-1) if parse.heads[i] == gold_heads[i]])

def transition(move, i, stack, parse):
    if move == SHIFT:
        stack.append(i)
        return i + 1
    elif move == RIGHT:
        parse.add(stack[-2], stack.pop())
        return i
    elif move == LEFT:
        parse.add(i, stack.pop())
        return i
    assert move in MOVES

def get_valid_moves(i, n, stack_depth):

    moves = []
    if (i+1) < n:
        moves.append(SHIFT)
    if stack_depth >= 2:
        moves.append(RIGHT)
    if stack_depth >= 1:
        move.append(LEFT)

    return moves

def get_gold_moves(n0, n, stack, heads, gold):

    def deps_between(target, others, gold):
        for word in others:
            if gold[word] == target or gold[target] == word:
                return True
        return False

    valid = get_valid_moves(n0, n, len(stack))
    if not stack or (SHIFT in valid and gold[n0] == stack[-1]):
        return [SHIFT]
    if gold[stack[-1]] == n0:
        return LEFT
    costly = set([m for m in MOVES if m not in valid])

    if len(stack) >= 2 and gold[stack[-1] == stack[-2]]:
        costly.add(LEFT)

    if SHIFT not in costly and deps_between(n0, stack, gold):
        costly.add(LEFT)
        costly.add(RIGHT)

    return [m for m in MOVES if m not in costly]

def extract_features(words, tags, n0, n, stack, parse):
    def get_stack_context(depth, stack, data):
        if depth >= 3:
            return data[stack[-1]], data[stack[-2]], data[stack[-3]]
        elif depth >= 2:
            return data[stack[-1]], data[stack[-2]], ''
        elif depth == 1:
            return data[stack[-1]], '', ''
        else:
            return '', '', ''

    def get_buffer_context(i, n, data):
        if i + 1 >=n:
            return data[i], '', ''
        elif i + 2 >= n:
            return data[i], data[i+1], ''
        else:
            return data[i], data[i+1], data[i+2]

    def get_parse_context(word, deps, data):
        if word == -1:
            return 0, '', ''
        deps = deps[word]
        valency = len(deps)
        if not valency:
            return 0, '', ''
        elif valency == 1:
            return 1, data[deps[-1]], ''
        else:
            return  valency, data[deps[-1]], data[deps[-2]]

    features = {}

