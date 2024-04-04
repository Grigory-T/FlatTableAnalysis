import textwrap
import math
import string
from typing import Iterable
import re

def wrap_text(string, max_width: int = 10) -> str:
    r"""
    used to draw node's label in graph. See FlatTableAnalysis.show_fd_graph()
    """
    return "\n".join(textwrap.wrap(string, max_width))

def powerset(n: int) -> int:
    r"""
    empty set not included, full set is included
    """
    if not n >= 1 or not isinstance(n, int):
        raise ValueError("argument `n` should be int >= 1")
    rv = 0
    for i in range(1, n+1):
        rv += math.comb(n, i)
    return rv

def cut_strings(
                list_of_strings: Iterable[str], 
                threshold: int=19
                ) -> Iterable[str]:
    r"""
    take list of strings
    cut them to minimal length that ensure uniqueness
    """
    if len(list_of_strings) != (len(set(list_of_strings))):
        raise ValueError("list_of_strings must be unique")

    max_string_len = max(map(len, list_of_strings))

    for i in range(max_string_len):
        cut_strings = set(string_[: i+1] for string_ in list_of_strings)
        if len(cut_strings) == len(list_of_strings):
            min_idx = i
            break

    min_idx = threshold if min_idx < threshold else min_idx
    result_list = [string_[: min_idx + 1] for string_ in list_of_strings]
    pat = re.compile(f'[{string.punctuation + string.whitespace}]+')
    result_list = [pat.sub("_", l) for l in result_list]
    return result_list
