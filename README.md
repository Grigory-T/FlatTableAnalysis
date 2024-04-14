

```python
!pip install flattableanalysis
from flattableanalysis.flat_table_analysis import FlatTableAnalysis

df = pd.DataFrame(YOUR_DATA)
fta = FlatTableAnalysis(df)  # create analysis object

fta.get_candidate_keys(2)  # check 2-cols candidates

fta.show_fd_graph()[0]  # see graph of functional dependencies (all pairs of columns)
```

# FlatTableAnalysis
Architectural analysis of a flat table. Discovering data from technical point of view

Article with description of the tool - https://habr.com/ru/articles/800473/
