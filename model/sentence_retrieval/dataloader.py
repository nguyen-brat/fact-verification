import collections
import glob
import logging
import os
import random
from typing import Dict, List, Tuple

BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])

class BiEncoderSample(object):
    query: str
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage]
    hard_negative_passages: List[BiEncoderPassage]