"""
Markov Chain Assignment


"""
from enum import Enum
from graph import MarkovChain
from graph import Node
import random
import pickle
import urllib
import urllib.request
import argparse
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--train", help = "train the iterator")
parser.add_argument("--input", help = "The input file to train on", default=sys.stdin)
parser.add_argument("--output", help = "The input file to train on", default=sys.stdout)
parser.add_argument("--word", help = "Use word tokenization")
parser.add_argument("--character", help = "Use character tokenization")
parser.add_argument("--byte", help = "Use byte tokenization")
parser.add_argument("--level", help = "Train or level n", type=int, default=1)
parser.add_argument("--generate", help = "generate an output")
parser.add_argument("--amount", help = "Generate n tokens", type=int)
parser.parse_args()
if parser.train:
    if parser.input:
        if parser.character:
            token = parser.character
        elif parser.byte:
            token = parser.byte
        else:
            token = parser.word
        if not parser.level:
            raise parser.error("Need a level")
        parser.train_url(parser.input)
    if parser.output:
        parser.save_pickle(parser.output)
if parser.generate:
    if not parser.amount:
        raise parser.error("Need an amount")
    if parser.input:
        print("I Tried my best :/")


class Tokenization(Enum):
    word = 1
    character = 2
    byte = 3
    none = 4


class RandomWriter(object):
    """A Markov chain based random data generator.
    """

    def __init__(self, level, tokenization=None):
        """Initialize a random writer.

        Args:
          level: The context length or "level" of model to build.
          tokenization: A value from Tokenization. This specifies how
            the data should be tokenized.

        The value given for tokenization will affect what types of
        data are supported.

        """
        self.graph = MarkovChain()
        self.level = level
        self.token = tokenization if tokenization else Tokenization.none
        self.state = None
        self.parent = None

    def generate(self):
        """Generate tokens using the model.

        Yield random tokens using the model. The generator should
        continue generating output indefinitely.

        It is possible for generation to get to a state that does not
        have any outgoing edges. You should handle this by selecting a
        new starting node at random and continuing.

        """
        # Generate a seed if one doesn't exist already
        if not self.state:
            self.state = random.choice(list(self.graph.nodes.values()))
            for token in self.state.name[:-1]:
                yield token
        while self.state:
            if not self.state.paths:
                yield self.state.name[-1]
                self.state = random.choice(list(self.graph.nodes.values()))
            else:
                path_dict = {path: (self.state.paths[path]/self.state.edge_count)
                             for path in self.state.paths}
                next_state = self.graph.nodes[self.random_pick(path_dict)]
                yield self.state.name[-1]
                if not next_state:
                    next_state = random.choice(list(self.graph.nodes.values()))
                self.state = next_state

    # Solution inspired by https://www.safaribooksonline.com/library/view/python-cookbook-2nd/0596007973/ch04s22.html
    def random_pick(self, path_dict):
        x = random.uniform(0, 1)
        cumulative_weight = 0.0
        for item, item_weight in path_dict.items():
            cumulative_weight += item_weight
            if x < cumulative_weight:
                return item

    def generate_file(self, filename, amount):
        """Write a file using the model.

        Args:
          filename: The name of the file to write output to.
          amount: The number of tokens to write.

        For character or byte tokens this should just output the
        tokens one after another. For any other type of token a space
        should be added between tokens. Use str to convert values to
        strings for printing.

        Do not duplicate any code from generate.

        Make sure to open the file in the appropriate mode.
        """
        fi = open(filename, 'w', encoding="utf-8")
        count = 0
        if self.token is Tokenization.byte or self.token is Tokenization.character:
            spacing = ""
        else:
            spacing = " "
        for generate_token in self.generate():
            count += 1
            write = str(generate_token)
            write += spacing
            fi.write(write)
            if count >= amount:
                break
        fi.close()

    def save_pickle(self, filename_or_file_object):
        """Write this model out as a Python pickle.

        Args:
          filename_or_file_object: A filename or file object to write
            to. You need to support both.

        If the argument is a file object you can assume it is opened
        in binary mode.

        """
        fi = open(filename_or_file_object, 'wb')
        with fi:
            pickle.dump(self, fi)
        fi.close()

    @classmethod
    def load_pickle(cls, filename_or_file_object):
        """Load a Python pickle and make sure it is in fact a model.

        Args:
          filename_or_file_object: A filename or file object to load
            from. You need to support both.
        Return:
          A new instance of RandomWriter which contains the loaded
          data.

        If the argument is a file object you can assume it is opened
        in binary mode.

        """
        fi = open(filename_or_file_object, 'rb')
        with fi:
            cls.graph = pickle.load(fi)
        fi.close()
        rw = cls(1)
        return rw

    def train_url(self, url):
        """Compute the probabilities based on the data downloaded from url.

        Args:
          url: The URL to download. Support any URL supported by
            urllib.

        This method is only supported if the tokenization mode is not
        none.

        Do not duplicate any code from train_iterable.

        """
        if self.token is not Tokenization.none:
            data = urllib.request.urlopen(url)
            data_input = data.read() if self.token is Tokenization.byte else data.read().decode("utf-8")
            self.train_iterable(data_input)

    def train_iterable(self, data):
        """Compute the probabilities based on the data given.

        If the tokenization mode is none, data must be an iterable. If
        the tokenization mode is character or word, then data must be
        a string. Finally, if the tokenization mode is byte, then data
        must be a bytes. If the type is wrong raise TypeError.

        Try to avoid storing all that data at one time, but if it is way
        simpler to store it don't worry about it. For most input types
        you will not need to store it.
        """
        # HINT: You will find you need to convert the input iterable
        # into a new iterable. One step is already implemented in the
        # final_tests.py files. You may use that code (making sure to
        # give credit where credit is due).

        if self.token == Tokenization.word or self.token == Tokenization.character:
            if not isinstance(data, str):
                raise TypeError("Data must be of type 'str'")
            if self.token is Tokenization.word:
                windows = self.windowed(data.split(), self.level)
            else:
                windows = self.windowed(list(data), self.level)

        if self.token == Tokenization.byte:
            if not isinstance(data, (int, bytes)):
                raise TypeError("Data must be of type int or bytes")
            windows = self.windowed(bytearray(data), self.level)

        if self.token == Tokenization.none:
            try:
                iterator_test = iter(data)
            except TypeError:
                return iterator_test
            windows = self.windowed(data, self.level)

        for window in windows:
            if window in self.graph.nodes:
                if self.parent:
                    self.parent.add_path(self.graph.nodes[window])
                    self.graph.update_node(self.parent)
                self.parent = self.graph.nodes[window]
            else:
                new_node = Node(window)
                self.graph.nodes[new_node.name] = new_node
                if self.parent:
                    self.parent.add_path(new_node)
                    self.graph.update_node(self.parent)
                self.parent = new_node

    # Taken from the smartest most handsome CS professor in the entire universe
    def windowed(self, iterable, size):
        """Convert an iterable to an iterable over a "windows" of the input.

        The windows are produced by sliding a window over the input iterable.
        """
        window = list()
        for v in iterable:
            if len(window) < size:
                window.append(v)
            else:
                window.pop(0)
                window.append(v)
            if len(window) == size:
                yield tuple(window)
