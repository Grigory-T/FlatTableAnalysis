import numpy as np
import pandas as pd

import itertools as it
import more_itertools as mit

import networkx as nx
import graphviz
import igraph as ig

import math
from collections.abc import Iterable
from collections import Counter
from typing import Union, Optional
from IPython.display import display
from tqdm.notebook import tqdm

from .utils import wrap_text


class FlatTableAnalysis:
    def __repr__(self) -> str:
        return f"FlatTableAnalysis instance\ndf.shape = {self.df.shape}"

    def __init__(
        self,
        df: pd.DataFrame,
        remove_constant_columns: bool=True,
        remove_all_unique_columns: bool=True,
        remove_one_one_relations: bool=True,
    ) -> None:
        r"""
        check df for basic validity
        modify df to internal representation (factorize, unify nan, ...)
        convert dtypes for speedup
        """
        if not isinstance(df.columns, pd.core.indexes.base.Index):
            raise ValueError("DataFrame header has more than one line")

        if not df.columns.is_unique:
            raise ValueError("DataFrame columns must be unique")

        if any(df.columns.isna()):
            raise ValueError("DataFrame columns must not contain NaN values")

        self.df = (
                    df.copy()
                    .replace(["None", "none", "nan", ""], [np.nan] * 4)
                    .fillna(np.nan)
                    .rename(str, axis=1)
                    .reset_index(drop=True)
        )

        # utils mappings
        self.col_pos = {col: pos for pos, col in enumerate(self.df)}.get
        self.col_to_unique = {
            col: self.df.loc[:, col].drop_duplicates().shape[0] for col in self.df
        }.get

        # remove columns
        if remove_constant_columns:
            to_delete = list(self.df.columns[self.df.columns.map(self.col_to_unique) == 1])
            self.df = self.df.loc[:, lambda df: df.columns.difference(to_delete)]
            print("remove_constant_columns: ", to_delete)
        if remove_all_unique_columns:
            to_delete = list(self.df.columns[self.df.columns.map(self.col_to_unique) == df.shape[0]])
            self.df = self.df.loc[:, lambda df: df.columns.difference(to_delete)]
            print("remove_all_unique_columns: ", to_delete)
        if remove_one_one_relations:
            self.df = self._delete_one_one_relations(self.df)

        self._df = (
            self.df
        )  # save df version without factorize, need for self.show_header_info()

        # final processing
        self.df = (
                    self.df.drop_duplicates()
                    .apply(lambda ser: ser.factorize()[0])
                    .pipe(lambda obj: obj if isinstance(obj, pd.DataFrame) else obj.to_frame())
                    .assign(
                                **{
                                    col: lambda df, col=col: df[col].astype("category")
                                    for col in self.df
                                }
                    )
        )

    def _delete_one_one_relations(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        r"""
        any pair of columns with strict one-to-one relation will be disturbing
        removing minimal number of columns to avoid this
        """
        # detext any pair of cols with one-to-one relation
        edge_list = []
        for cols in it.combinations(df, r=2):
            if (
                self.col_to_unique(cols[0])
                == self.col_to_unique(cols[1])
                == df.loc[:, cols].drop_duplicates().shape[0]
            ):
                edge_list.append(cols)

        # if e.g. 4 cols have one-to-one relations between them -> need do delete all but one from them
        G = nx.Graph()
        G.add_edges_from(edge_list)
        ccs = [sorted(cc, key=self.col_pos) for cc in nx.connected_components(G)]
        to_delete = [cc[1:] for cc in ccs]
        to_delete = list(it.chain.from_iterable(to_delete))
        if to_delete:
            print("remove_one_one_relations: ", to_delete)
            print("    found these sets of one-one relations, keep only 1st item from each: ", ccs)
            return df.drop(to_delete, axis=1)
        else:
            return df

    def show_header_info(self) -> None:
        r"""
        print header and relevant info about each columns. useful for analysis overview
        """
        print(
            f'{"idx":<5}',
            f'{"col name":<15}',
            f'{"unique count":<15}',
            f'{"nan count":<15}',
            f'{"dtype":<15}',
            f'{"examples":<15}',
        )
        for idx, col in enumerate(self.df):
            print(
                f"{idx:<5}",
                f"{col:<20}"[:18],
                f"{self.col_to_unique(col):<15_}",
                f"{sum(self._df[col].isna()):<15_}",
                f"{str(self._df[col].dtype):<15}",
                f"{str(list(self._df[col].unique()[:5])):<15}"[:50],
            )
        print(f"total rows: {self.df.shape[0]:_}")

    def get_candidate_keys(
        self,
        col_nums: Optional[Union[int, Iterable[int]]] = 1,
        strict: bool=False
    ) -> pd.DataFrame:
        r"""
        calculate cardinality for all sets of cols with given lengths
        sort them to show bigger cardinality and smaller sets of cols
        """
        col_nums = list(col_nums) if isinstance(col_nums, Iterable) else [col_nums]
        if max(col_nums) > self.df.shape[1]:
            raise ValueError("Maximum number of columns specified is larger than the number of DataFrame columns")
        if min(col_nums) <= 0:
            raise ValueError("Minimum number of columns must be greater than 0")

        pbar = tqdm(
            total=sum(math.comb(self.df.shape[1], col_num) for col_num in col_nums)
        )
        result = []
        for col_num in col_nums:
            for col_names in it.combinations(self.df, r=col_num):
                result.append(
                    (col_names, self.df.loc[:, col_names].drop_duplicates().shape[0])
                )
                pbar.update(1)
        pbar.close()

        if not strict:
            return  (
                pd.DataFrame(result, columns=["col_names", "uniques"])
                .pipe(
                    lambda df: df.insert(1, "col_names_len", df["col_names"].str.len())
                    or df
                )
                .pipe(
                    lambda df: df.assign(
                        col_names=lambda df: df["col_names"].map(
                            lambda el: tuple(sorted(el, key=self.col_pos))
                        )
                    )
                )
                .assign(total_rows=self.df.shape[0])
                .assign(uniques_frac=lambda df: df["uniques"] / df["total_rows"])
                .assign(col_names_pos = lambda df: 
                        df['col_names'].map(lambda el: tuple(map(self.col_pos, el)))  # for sorting only
                        )
                .sort_values(["uniques_frac", "col_names_len", 'col_names_pos'], ascending=[False, True, True])
                .drop('col_names_pos', axis=1)
                .reset_index(drop=True)
            )
        
        elif strict:
            potential_keys = [set(el[0]) for el in result if el[1] == self.df.shape[0]]
            candidate_keys = []
            for potential_key in potential_keys:
                flag = True
                for candidate_key in candidate_keys:
                    if candidate_key.issubset(potential_key):
                        flag = False
                        break
                if flag:
                    candidate_keys.append(potential_key)
            
            candidate_keys = [[el] for el in candidate_keys]  # nested list -> resulting in one column in df
            return  (
                pd.DataFrame(candidate_keys, columns=["col_names"])
                .pipe(
                        lambda df: df.assign(
                            col_names=lambda df: df["col_names"].map(
                                lambda el: tuple(sorted(el, key=self.col_pos))
                        )
                    )
                )
                .assign(col_names_len = lambda df: df['col_names'].str.len())
                .assign(col_names_pos = lambda df: 
                        df['col_names'].map(lambda el: tuple(map(self.col_pos, el)))  # for sorting only
                        )
                .sort_values(["col_names_len", 'col_names_pos'], ascending=[True, True])
                .drop('col_names_pos', axis=1)
                .reset_index(drop=True)
            )

    def show_fd_graph(
        self,
        threshold: Optional[Union[float, int]] = 1,
    ) -> graphviz.Digraph:
        r"""
        draw graph with functional dependancies (fds) as edges (between each pair of columns)
        fds may be not strict (threshold level)
        for better visualization we remove transitive dependancies
        """
        if not (0 < threshold <= 1):
            raise ValueError("Threshold should be in the left-open interval (0, 1]")

        table = []
        for cols in it.combinations(self.df, r=2):
            table.append(
                (cols[0], cols[1], self.df.loc[:, cols].drop_duplicates().shape[0])
            )

        table = (
            pd.DataFrame(table, columns=["col_L", "col_R", "unique_LR"])
            .assign(unique_L=lambda df: df["col_L"].map(self.col_to_unique))
            .assign(unique_R=lambda df: df["col_R"].map(self.col_to_unique))
            .assign(frac_L=lambda df: df["unique_L"] / df["unique_LR"])
            .assign(frac_R=lambda df: df["unique_R"] / df["unique_LR"])
        )

        edge_list = []
        for _, (col_L, col_R, *__, frac_L, frac_R) in table.iterrows():
            if frac_L >= threshold:
                edge_list.append((col_L, col_R, {"weight": round(frac_L, 2)}))
            if frac_R >= threshold:
                edge_list.append((col_R, col_L, {"weight": round(frac_R, 2)}))

        G = nx.DiGraph()
        G.add_edges_from(edge_list)
        if mit.first(nx.simple_cycles(G), False):  # if at leaste one cycle exists
            H = ig.Graph.from_networkx(G)
            edges_to_remove = H.feedback_arc_set()
            H.delete_edges(edges_to_remove)
            print(f"simple cylces removed {len(edges_to_remove)}")
            G = H.to_networkx()
        G_tr = nx.transitive_reduction(G)
        # need to copy data for nodes and edges manually
        G_tr.add_nodes_from(G.nodes(data=True))
        G_tr.add_edges_from((u, v, G.edges[u, v]) for u, v in G_tr.edges)

        K = graphviz.Digraph(node_attr={"shape": "box"})
        for col in self.df:
            K.node(wrap_text(col))
        for L, R, data in G_tr.edges(data=True):
            K.edge(wrap_text(L), wrap_text(R), label=str(data["weight"]))
        return K

    def get_density_table(self) -> pd.DataFrame:
        r"""
        applied to all columns pairs
        pair consists two columns
        main info: count of links compared to min and max possible (density)
        """
        result = []
        for cols in it.combinations(self.df, r=2):
            d = {}
            d["left_columns"] = cols[0]
            d["right_columns"] = cols[1]
            d["total_unique"] = self.df.loc[:, cols].drop_duplicates().shape[0]
            d["left_side_unique"] = self.col_to_unique(cols[0])
            d["right_side_unique"] = self.col_to_unique(cols[1])
            result.append(d)
        return (
            pd.DataFrame(result)
            .assign(
                density=lambda df: df["total_unique"]
                / (df["left_side_unique"] * df["right_side_unique"])
            )
            .sort_values("density", ascending=False)
            .reset_index(drop=True)
        )

    def show_set_relation(
        self,
        L: Optional[Union[str, Iterable[str]]] = None,
        R: Optional[Union[str, Iterable[str]]] = None,
    ) -> None:
        r"""
        deep analysis of pair
        pair consists of two objects. each object may be one column or set of columns
        main info: types of connected components and their count

        additionaly shows tails types (nan, blanks, void) for each side of pair
        """
        L = L or self.df.columns[0]
        R = R or self.df.columns[1]
        L = [L] if isinstance(L, str) else list(L)
        R = [R] if isinstance(R, str) else list(R)

        L_n, R_n = len(L), len(R)

        G = nx.Graph()
        L = self.df.loc[:, L].values.tolist()
        R = self.df.loc[:, R].values.tolist()
        for L_tup, R_tup in zip(L, R):
            L_tup = (0,) + tuple(L_tup)
            R_tup = (1,) + tuple(R_tup)
            G.add_edge(L_tup, R_tup)

        CC_type = Counter()
        CC_all = []
        for cc in nx.connected_components(G):
            cc_flag = [node[0] for node in cc]
            L_count = cc_flag.count(0)
            R_count = cc_flag.count(1)
            CC_all.append((L_count, R_count))
            if L_count > 1 and R_count > 1:
                CC_type["many_many"] += 1
            elif L_count > 1:
                CC_type["many_one"] += 1
            elif R_count > 1:
                CC_type["one_many"] += 1
            elif L_count == 1 and R_count == 1:
                CC_type["one_one"] += 1

        result = pd.Series(CC_type).reset_index().set_axis(["CC_type", "count"], axis=1)

        CC_all = (
            sorted(CC_all, key=lambda el: el[0], reverse=True)[:20]
            + (["..."] if len(CC_all) > 20 else [])
            + sorted(CC_all, key=lambda el: el[1], reverse=True)[:20]
            + (["..."] if len(CC_all) > 20 else [])
        )

        zero_symbol = -1
        try:
            L_nan_cc = nx.node_connected_component(G, tuple([0] + [zero_symbol] * L_n))
        except KeyError:
            L_nan_cc = []
        cc_flag = [node[0] for node in L_nan_cc]
        L_nan_L_count = cc_flag.count(0)
        L_nan_R_count = cc_flag.count(1)

        try:
            R_nan_cc = nx.node_connected_component(G, tuple([1] + [zero_symbol] * R_n))
        except KeyError:
            R_nan_cc = []
        cc_flag = [node[0] for node in R_nan_cc]
        R_nan_L_count = cc_flag.count(0)
        R_nan_R_count = cc_flag.count(1)

        display(result)
        print("_" * 50)
        display(CC_all)
        print("_" * 50)
        display(
            (
                L_nan_L_count,
                L_nan_R_count,
                R_nan_L_count,
                R_nan_R_count,
                L_nan_cc == R_nan_cc,
            )
        )

    def show_opposite_count(
        self,
        L: Optional[Union[str, Iterable[str]]] = None,
        R: Optional[Union[str, Iterable[str]]] = None,
    ) -> None:
        r"""
        additional analysis of pair
        pair consists of two objects
        each object may be columns or set of columns
        main info: breakdown of elements in the object by links count (=opposite side count)
        """
        L = L or self.df.columns[0]
        R = R or self.df.columns[1]
        L = [L] if isinstance(L, str) else list(L)
        R = [R] if isinstance(R, str) else list(R)

        sub_df = self.df.loc[:, L + R].drop_duplicates()
        L_dict = sub_df.loc[:, L].value_counts().value_counts().to_dict()
        R_dict = sub_df.loc[:, R].value_counts().value_counts().to_dict()

        L_df = (
            pd.Series(L_dict)
            .reset_index()
            .set_axis(["OPPOSITE " + ", ".join(R), "THIS " + ", ".join(L)], axis=1)
            .iloc[:, [1, 0]]
            .pipe(lambda df: df.sort_values(df.columns[-1], ascending=False))
        )
        R_df = (
            pd.Series(R_dict)
            .reset_index()
            .set_axis(["OPPOSITE " + ", ".join(L), "THIS " + ", ".join(R)], axis=1)
            .iloc[:, [1, 0]]
            .pipe(lambda df: df.sort_values(df.columns[-1], ascending=False))
        )
        display(L_df)
        display(R_df)

    def get_cols_determinants(
        self,
        target: Optional[Union[str, Iterable[str]]] = None,
        col_nums: Optional[Union[int, Iterable[int]]] = 1,
    ) -> pd.DataFrame:
        r"""
        deep analysis of possible determinants of the object
        object may be one columns or set of columns
        determinants may be one column or set of columns
        main info: what columns defines object the best
        """
        target = target or [self.df.columns[0]]
        target = (str(target),) if isinstance(target, (int, str)) else tuple(target)

        col_nums = [col_nums] if isinstance(col_nums, int) else list(col_nums)
        if max(col_nums) > self.df.shape[1] - len(target):
            raise ValueError("Maximum number of columns specified is larger than the number of available DataFrame columns")
        if min(col_nums) <= 0:
            raise ValueError("Minimum number of columns must be greater than 0")

        other_cols = list(self.df.columns.difference(target))

        pbar = tqdm(
            total=sum(math.comb(len(other_cols), col_num) for col_num in col_nums)
        )
        result = []
        for col_num in col_nums:
            for source in it.combinations(other_cols, r=col_num):
                sub_df = self.df.loc[:, source + target].drop_duplicates()
                LR_unique = sub_df.loc[:, source + target].drop_duplicates().shape[0]
                L_unique = sub_df.loc[:, source].drop_duplicates().shape[0]
                result.append((source, L_unique / LR_unique))
                pbar.update(1)
        pbar.close

        return (
            pd.DataFrame(result, columns=["cols", "ratio"])
            .sort_values("ratio", ascending=False)
            .reset_index(drop=True)
        )
