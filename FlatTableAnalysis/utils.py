import textwrap


def wrap_text(string, max_width: int = 10) -> str:
    r"""
    used to draw node's label in graph. See FlatTableAnalysis.show_fd_graph()
    """
    return "\n".join(textwrap.wrap(string, max_width))
