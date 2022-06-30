"""
Modul sadrži implementaciju stabla.
"""
import math

from queue import Queue


class TreeNode(object):
    __slots__ = 'parent', 'children', 'data'

    def __init__(self, data):
        self.parent = None
        self.children = []
        self.data = data

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.children) == 0

    def add_child(self, x):
        x.parent = self
        self.children.append(x)

    def set_data(self,data):
        self.data = data


class Tree(object):
    def __init__(self):
        self.root = None

    def is_empty(self):
        return self.root is None

    def depth(self, x):
        if x.is_root():
            return 0
        else:
            return 1 + self.depth(x.parent)

    def _height(self, x):
        if x.is_leaf():
            return 0
        else:
            return 1 + max(self._height(c) for c in x.children)

    def height(self):
        return self._height(self.root)

    def preorder(self, x):
        if not self.is_empty():
            for c in x.children:
                self.preorder(c)

    def postorder(self, x):
        if not self.is_empty():
            for c in x.children:
                self.postorder(c)

    def breadth_first(self):
        to_visit = Queue()
        to_visit.enqueue(self.root)
        while not to_visit.is_empty():
            e = to_visit.dequeue()

            for c in e.children:
                to_visit.enqueue(c)


def pprint_tree(node, file=None, _prefix="", _last=True):
    print(_prefix, "`- " if _last else "|- ", node.data, sep="", file=file)
    _prefix += "   " if _last else "|  "
    child_count = len(node.children)
    for i, child in enumerate(node.children):
        _last = i == (child_count - 1)
        pprint_tree(child, file, _prefix, _last)

def print_first_children(node):
    for i in node.children:
        print(i.data,end=',')