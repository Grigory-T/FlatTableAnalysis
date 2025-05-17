import pandas as pd
import graphviz
import networkx as nx

from collections.abc import Iterable
from typing import Union, Optional, Tuple, List

from .preprocessing import preprocess_dataframe
from .analysis import (
    get_candidate_keys as _get_candidate_keys,
    show_fd_graph as _show_fd_graph,
    get_density_table as _get_density_table,
    show_set_relation as _show_set_relation,
    fds as _fds,
    show_opposite_count as _show_opposite_count,
    get_cols_determinants as _get_cols_determinants,
)


class FlatTableAnalysis:
    def __repr__(self) -> str:
        return f"FlatTableAnalysis instance\ndf.shape = {self.df.shape}"

    def __init__(
        self,
        df: pd.DataFrame,
        remove_constant_columns: bool = True,
        remove_all_unique_columns: bool = True,
        remove_one_one_relations: bool = True,
    ) -> None:
        self.df = preprocess_dataframe(
            df,
            remove_constant_columns=remove_constant_columns,
            remove_all_unique_columns=remove_all_unique_columns,
            remove_one_one_relations=remove_one_one_relations,
        )

        self.col_pos = {col: pos for pos, col in enumerate(self.df)}.get
        self.col_to_unique = {
            col: self.df.loc[:, col].drop_duplicates().shape[0] for col in self.df
        }.get

        self._make_header_info()

        self.df = (
            self.df.drop_duplicates()
            .apply(lambda ser: ser.factorize()[0])
            .pipe(lambda obj: obj if isinstance(obj, pd.DataFrame) else obj.to_frame())
            .assign(
                **{col: lambda df, col=col: df[col].astype("category") for col in self.df}
            )
        )

    def _make_header_info(self) -> None:
        col_name_width = min(max(len(c) for c in self.df) + 5, 35)
        self.header_info = {}
        d = {}
        d["idx"] = "idx".ljust(5)
        d["col name"] = "col name".ljust(col_name_width)
        d["unique count"] = "unique count".ljust(15)
        d["nan count"] = "nan count".ljust(15)
        d["dtype"] = "dtype".ljust(15)
        d["examples"] = "examples".ljust(15)
        self.header_info["header"] = d
        for idx, col in enumerate(self.df):
            d = {}
            d["idx"] = f"{idx:<5}"
            d["col name"] = col[: col_name_width - 3].ljust(col_name_width)
            d["unique count"] = f"{self.col_to_unique(col):<15_}"
            d["nan count"] = f"{sum(self.df[col].isna()):<15_}"
            d["dtype"] = str(self.df[col].dtype).ljust(15)
            d["examples"] = str(list(self.df[col].dropna().unique()[:5]))[:70]
            self.header_info[col] = d

    def show_header_info(self) -> None:
        for _, header_info in self.header_info.items():
            print(
                header_info["idx"],
                header_info["col name"],
                header_info["unique count"],
                header_info["nan count"],
                header_info["dtype"],
                header_info["examples"],
            )
        print(f"total rows: {self.df.shape[0]:_}")

    def get_candidate_keys(self, col_nums: int = 1) -> pd.DataFrame:
        return _get_candidate_keys(self.df, self.col_pos, col_nums)

    def show_fd_graph(
        self,
        threshold: Optional[Union[float, int]] = 1,
    ) -> Tuple[graphviz.Digraph, nx.classes.digraph.DiGraph]:
        return _show_fd_graph(self.df, self.col_to_unique, threshold)

    def get_density_table(self) -> pd.DataFrame:
        return _get_density_table(self.df, self.col_to_unique)

    def show_set_relation(
        self,
        L: Optional[Union[str, Iterable[str]]] = None,
        R: Optional[Union[str, Iterable[str]]] = None,
        level: int = 1,
    ) -> None:
        _show_set_relation(self.df, L, R, level)

    def fds(self, data: Iterable[Iterable[Iterable[str]]]) -> List[Tuple[bool, bool]]:
        return _fds(self.df, data)

    def show_opposite_count(
        self,
        L: Optional[Union[str, Iterable[str]]] = None,
        R: Optional[Union[str, Iterable[str]]] = None,
    ) -> None:
        _show_opposite_count(self.df, L, R)

    def get_cols_determinants(
        self,
        target: Optional[Union[str, Iterable[str]]] = None,
        max_cols: int = 1,
    ) -> pd.DataFrame:
        return _get_cols_determinants(self.df, self.col_pos, target, max_cols)
