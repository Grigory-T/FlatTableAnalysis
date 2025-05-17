import math
import itertools as it
from collections import Counter
from collections.abc import Iterable
from typing import Iterable as TypingIterable, Optional, Union, Tuple, List

import pandas as pd
import networkx as nx
import graphviz
from IPython.display import display
from tqdm.notebook import tqdm

from .utils import wrap_text


def get_candidate_keys(
    df: pd.DataFrame,
    col_pos,
    col_nums: int = 1,
) -> pd.DataFrame:
    if col_nums > df.shape[1]:
        raise ValueError(
            "number of columns specified is larger than the number of DataFrame columns"
        )
    if col_nums <= 0:
        raise ValueError("number of columns must be greater than 0")

    pbar = tqdm(
        total=sum(math.comb(df.shape[1], col_num) for col_num in range(1, col_nums + 1))
    )
    candidates = []
    for col_num in range(1, col_nums + 1):
        for col_names in it.combinations(df, r=col_num):
            unique_n = sum(~df.duplicated(subset=col_names))
            col_names_set = set(col_names)

            flag = True
            for candidate in candidates:
                if col_names_set > candidate[0] and unique_n <= candidate[1]:
                    flag = False
                    break

            if flag:
                candidates.append((col_names_set, unique_n))
            pbar.update(1)
    pbar.close()

    return (
        pd.DataFrame(candidates, columns=["col_names", "uniques"])
        .assign(col_names=lambda df_: df_["col_names"].map(lambda el: tuple(sorted(el, key=col_pos))))
        .assign(total_rows=df.shape[0])
        .assign(col_names_len=lambda df_: df_["col_names"].str.len())
        .assign(uniques_frac=lambda df_: df_["uniques"] / df_["total_rows"])
        .assign(col_names_pos=lambda df_: df_["col_names"].map(lambda el: tuple(map(col_pos, el))))
        .sort_values(["uniques_frac", "col_names_len", "col_names_pos"], ascending=[False, True, True])
        .drop("col_names_pos", axis=1)
        .reset_index(drop=True)
        .loc[:, ["col_names", "col_names_len", "uniques", "total_rows", "uniques_frac"]]
    )


def show_fd_graph(
    df: pd.DataFrame,
    col_to_unique,
    threshold: Optional[Union[float, int]] = 1,
) -> Tuple[graphviz.Digraph, nx.classes.digraph.DiGraph]:
    if not 0 < threshold <= 1:
        raise ValueError("Threshold should be in the left-open interval (0, 1]")

    table = []
    for cols in it.combinations(df, r=2):
        table.append((cols[0], cols[1], df.loc[:, cols].drop_duplicates().shape[0]))

    table = (
        pd.DataFrame(table, columns=["col_L", "col_R", "unique_LR"])
        .assign(unique_L=lambda df_: df_["col_L"].map(col_to_unique))
        .assign(unique_R=lambda df_: df_["col_R"].map(col_to_unique))
        .assign(frac_L=lambda df_: df_["unique_L"] / df_["unique_LR"])
        .assign(frac_R=lambda df_: df_["unique_R"] / df_["unique_LR"])
    )

    edge_list = []
    for _, (col_L, col_R, *__, frac_L, frac_R) in table.iterrows():
        if frac_L >= threshold:
            edge_list.append((col_L, col_R, {"weight": frac_L}))
        if frac_R >= threshold:
            edge_list.append((col_R, col_L, {"weight": frac_R}))

    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    if threshold == 1:
        G_tr = nx.transitive_reduction(G)
        G_tr.add_nodes_from(G.nodes(data=True))
        G_tr.add_edges_from((u, v, G.edges[u, v]) for u, v in G_tr.edges)
        G = G_tr

    K = graphviz.Digraph()
    K.attr(nodesep=".3", ranksep=".3", rankdir="TB", bgcolor="antiquewhite", fontsize="10")
    K.attr("node", shape="box", style="filled", color="lightblue2")
    for col in df:
        K.node(wrap_text(col).replace(':', '_'))
    for L, R, data in G.edges(data=True):
        K.edge(wrap_text(L).replace(':', '_'), wrap_text(R).replace(':', '_'), label=str(data["weight"]))
    K = K.unflatten(stagger=3, fanout=True, chain=5)
    return K, G


def get_density_table(df: pd.DataFrame, col_to_unique) -> pd.DataFrame:
    result = []
    for cols in it.combinations(df, r=2):
        d = {
            "left_columns": cols[0],
            "right_columns": cols[1],
            "total_unique": df.loc[:, cols].drop_duplicates().shape[0],
            "left_side_unique": col_to_unique(cols[0]),
            "right_side_unique": col_to_unique(cols[1]),
        }
        result.append(d)
    return (
        pd.DataFrame(result)
        .assign(density=lambda df_: df_["total_unique"] / (df_["left_side_unique"] * df_["right_side_unique"]))
        .sort_values("density", ascending=False)
        .reset_index(drop=True)
    )


def show_set_relation(
    df: pd.DataFrame,
    L: Optional[Union[str, TypingIterable[str]]] = None,
    R: Optional[Union[str, TypingIterable[str]]] = None,
    level: int = 1,
) -> None:
    L = L or df.columns[0]
    R = R or df.columns[1]
    L = [L] if isinstance(L, str) else list(L)
    R = [R] if isinstance(R, str) else list(R)

    subdf = df.loc[:, L + R].drop_duplicates()
    total = subdf.shape[0]
    L_one_nodes = sum(~subdf.loc[:, L].duplicated(keep=False))
    R_one_nodes = sum(~subdf.loc[:, R].duplicated(keep=False))
    L_nodes = sum(~subdf.loc[:, L].duplicated(keep="first"))
    R_nodes = sum(~subdf.loc[:, R].duplicated(keep="first"))
    LR_frac_edges = L_one_nodes / total
    RL_frac_edges = R_one_nodes / total
    LR_frac_nodes = L_one_nodes / L_nodes
    RL_frac_nodes = R_one_nodes / R_nodes
    print(
        f"left unique {L_nodes:_}, right unique {R_nodes:_}, edges {total:_} ({total / (L_nodes * R_nodes)}%)"
    )
    print(f"nodes: left fd -> {LR_frac_nodes}, right fd -> {RL_frac_nodes}")
    print(f"edges: left fd -> {LR_frac_edges}, right fd -> {RL_frac_edges}")
    if level == 1:
        return

    G = nx.Graph()
    L_list = df.loc[:, L].values.tolist()
    R_list = df.loc[:, R].values.tolist()
    for L_tup, R_tup in zip(L_list, R_list):
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

    result = (
        pd.Series(CC_type)
        .reset_index()
        .set_axis(["CC_type", "count"], axis=1)
        .sort_values("CC_type")
    )
    display(result)
    if level == 2:
        return

    CC_all = (
        sorted(CC_all, key=lambda el: el[0], reverse=True)[:20]
        + (["..."] if len(CC_all) > 20 else [])
        + sorted(CC_all, key=lambda el: el[1], reverse=True)[:20]
        + (["..."] if len(CC_all) > 20 else [])
    )

    zero_symbol = -1
    try:
        L_nan_cc = nx.node_connected_component(G, tuple([0] + [zero_symbol] * len(L)))
    except KeyError:
        L_nan_cc = []
    cc_flag = [node[0] for node in L_nan_cc]
    L_nan_L_count = cc_flag.count(0)
    L_nan_R_count = cc_flag.count(1)

    try:
        R_nan_cc = nx.node_connected_component(G, tuple([1] + [zero_symbol] * len(R)))
    except KeyError:
        R_nan_cc = []
    cc_flag = [node[0] for node in R_nan_cc]
    R_nan_L_count = cc_flag.count(0)
    R_nan_R_count = cc_flag.count(1)

    print("_" * 50)
    display(CC_all)
    print("_" * 50)
    display((L_nan_L_count, L_nan_R_count, R_nan_L_count, R_nan_R_count, L_nan_cc == R_nan_cc))
    if level == 3:
        return


def fds(
    df: pd.DataFrame,
    data: Iterable[TypingIterable[TypingIterable[str]]],
) -> List[Tuple[bool, bool]]:
    rv = []
    for L, R in tqdm(data):
        subdf = df.loc[:, L + R].drop_duplicates()
        rv.append((any(subdf.loc[:, L].duplicated()), any(subdf.loc[:, R].duplicated())))
    return rv


def show_opposite_count(
    df: pd.DataFrame,
    L: Optional[Union[str, TypingIterable[str]]] = None,
    R: Optional[Union[str, TypingIterable[str]]] = None,
) -> None:
    L = L or df.columns[0]
    R = R or df.columns[1]
    L = [L] if isinstance(L, str) else list(L)
    R = [R] if isinstance(R, str) else list(R)

    sub_df = df.loc[:, L + R].drop_duplicates()
    L_dict = sub_df.loc[:, L].value_counts().value_counts().to_dict()
    R_dict = sub_df.loc[:, R].value_counts().value_counts().to_dict()

    L_df = (
        pd.Series(L_dict)
        .reset_index()
        .set_axis(["OPPOSITE " + ", ".join(R), "THIS " + ", ".join(L)], axis=1)
        .iloc[:, [1, 0]]
        .pipe(lambda df_: df_.sort_values(df_.columns[-1], ascending=False))
    )
    R_df = (
        pd.Series(R_dict)
        .reset_index()
        .set_axis(["OPPOSITE " + ", ".join(L), "THIS " + ", ".join(R)], axis=1)
        .iloc[:, [1, 0]]
        .pipe(lambda df_: df_.sort_values(df_.columns[-1], ascending=False))
    )
    display(L_df)
    display(R_df)


def get_cols_determinants(
    df: pd.DataFrame,
    col_pos,
    target: Optional[Union[str, TypingIterable[str]]] = None,
    max_cols: int = 1,
) -> pd.DataFrame:
    target = target or [df.columns[0]]
    target = (str(target),) if isinstance(target, (int, str)) else tuple(target)

    if max_cols + len(target) > df.shape[1]:
        raise ValueError(
            "Maximum number of columns specified is larger than the number of available DataFrame columns"
        )
    if max_cols <= 0:
        raise ValueError("Minimum number of columns must be greater than 0")

    other_cols = list(df.columns.difference(target))

    pbar = tqdm(
        total=sum(math.comb(len(other_cols), col_num) for col_num in range(1, max_cols + 1))
    )

    determinants = []
    for col_num in range(1, max_cols + 1):
        for source in it.combinations(other_cols, r=col_num):
            subdf = df.loc[:, source + target].drop_duplicates()
            unique_n = sum(~subdf.loc[:, source].duplicated(keep=False))
            frac = unique_n / subdf.shape[0]
            source_set = set(source)

            flag = True
            for determinant in determinants:
                if source_set > determinant[0] and frac <= determinant[1]:
                    flag = False
                    break

            if flag:
                determinants.append((source_set, frac))
            pbar.update(1)
    pbar.close()

    return (
        pd.DataFrame(determinants, columns=["col_names", "frac"])
        .assign(col_names=lambda df_: df_["col_names"].map(lambda el: tuple(sorted(el, key=col_pos))))
        .assign(col_names_len=lambda df_: df_["col_names"].str.len())
        .loc[:, ["col_names", "col_names_len", "frac"]]
        .assign(col_names_pos=lambda df_: df_["col_names"].map(lambda el: tuple(map(col_pos, el))))
        .sort_values(["frac", "col_names_len", "col_names_pos"], ascending=[False, True, True])
        .drop("col_names_pos", axis=1)
        .reset_index(drop=True)
    )
