import textwrap
import math
import string
from typing import Iterable, Union
import re
import pandas as pd


def wrap_text(s, max_width: int = 10) -> str:
    r"""
    used to draw node's label in graph. See FlatTableAnalysis.show_fd_graph()
    """
    return "\n".join(textwrap.wrap(s, max_width))


def powerset(n: int, i_max=None) -> int:
    r"""
    empty set not included, full set is included
    """
    i_max = i_max or n
    if not n >= 1 or not isinstance(n, int):
        raise ValueError("argument `n` should be int >= 1")
    rv = 0
    for i in range(1, i_max + 1):
        rv += math.comb(n, i)
    return rv


def cut_strings(list_of_strings: Iterable[str], threshold: int = 19) -> Iterable[str]:
    r"""
    take list of strings
    cut them to minimal length that ensure uniqueness
    """
    if len(list_of_strings) != (len(set(list_of_strings))):
        raise ValueError("list_of_strings must be unique")

    pat = re.compile(f"[{string.punctuation + string.whitespace}]+")
    list_of_strings = [pat.sub("_", l) for l in list_of_strings]
    list_of_strings = make_unique(list_of_strings)

    max_string_len = max(map(len, list_of_strings))

    for i in range(max_string_len):
        cut_strings_set = set(string_[: i + 1] for string_ in list_of_strings)
        if len(cut_strings_set) == len(list_of_strings):
            min_idx = i
            break

    min_idx = threshold if min_idx < threshold else min_idx
    result_list = [string_[: min_idx + 1] for string_ in list_of_strings]
    return result_list


def make_unique(list_of_el: Iterable[Union[str, int]]) -> pd.core.series.Series:
    r"""
    take iterable of elements
    if there are duplicates - add consecutive numbering to them
    it ensures uniqueness of elements
    """
    list_of_el = [str(el) for el in list_of_el]
    return (
        pd.DataFrame(list_of_el, columns=["input_string"])
        .assign(string_cumcount=lambda df: df.groupby("input_string").agg("cumcount"))
        .assign(duplicated_string=lambda df: df["input_string"].duplicated(keep=False))
        .apply(
            lambda el: (
                el["input_string"] + "_" + str(el["string_cumcount"])
                if el["duplicated_string"]
                else el["input_string"]
            ),
            axis=1,
        )
    )
