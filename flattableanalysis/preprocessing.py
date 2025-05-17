import itertools as it
import numpy as np
import pandas as pd
import networkx as nx


def delete_one_one_relations(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that have strict one-to-one relations."""
    col_pos = {col: pos for pos, col in enumerate(df)}
    col_to_unique = {col: df[col].drop_duplicates().shape[0] for col in df}

    edge_list = []
    for cols in it.combinations(df, r=2):
        if (
            col_to_unique[cols[0]]
            == col_to_unique[cols[1]]
            == df.loc[:, cols].drop_duplicates().shape[0]
        ):
            edge_list.append(cols)

    G = nx.Graph()
    G.add_edges_from(edge_list)
    ccs = [sorted(cc, key=col_pos.get) for cc in nx.connected_components(G)]
    to_delete = list(it.chain.from_iterable(cc[1:] for cc in ccs))
    if to_delete:
        print("remove_one_one_relations: ", to_delete)
        print("    found these sets of one-one relations, keep only 1st item from each: ", ccs)
        df = df.drop(to_delete, axis=1)
    return df


def preprocess_dataframe(
    df: pd.DataFrame,
    remove_constant_columns: bool = True,
    remove_all_unique_columns: bool = True,
    remove_one_one_relations: bool = True,
) -> pd.DataFrame:
    """Validate and clean DataFrame used for FlatTableAnalysis."""
    if not isinstance(df.columns, pd.core.indexes.base.Index):
        raise ValueError("DataFrame header has more than one line")

    if not df.columns.is_unique:
        raise ValueError("DataFrame columns must be unique")

    if any(df.columns.isna()):
        raise ValueError("DataFrame columns must not contain NaN values")

    df = (
        df.replace(["None", "none", "nan", ""], [np.nan] * 4)
        .fillna(np.nan)
        .rename(str, axis=1)
        .reset_index(drop=True)
    )

    col_to_unique = {col: df[col].drop_duplicates().shape[0] for col in df}

    if remove_constant_columns:
        to_delete = list(df.columns[df.columns.map(lambda c: col_to_unique[c]) == 1])
        df = df.loc[:, lambda d: d.columns.difference(to_delete)]
        print("remove_constant_columns: ", to_delete)

    if remove_all_unique_columns:
        to_delete = list(
            df.columns[df.columns.map(lambda c: col_to_unique[c]) == df.shape[0]]
        )
        df = df.loc[:, lambda d: d.columns.difference(to_delete)]
        print("remove_all_unique_columns: ", to_delete)

    if remove_one_one_relations:
        df = delete_one_one_relations(df)

    if df.empty:
        raise Exception("removed all columns during preprocessing")

    return df
