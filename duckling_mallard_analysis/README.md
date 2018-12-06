# Files

**sys_queries.txt**: Contains queries with original labels

**sys_queries_clean.txt**: Queries with only text no labels

**duckling_results.p, mallard_results.p**: Parsed output from Duckling/Mallard for each query. Each query has a corresponding list of tuples, 
one for each entity in the form (start_index, end_index, label). Saved a pickle for faster analysis.


# Code (testing.py)§

Correct in this context means that every entity is detected with the exact span.

**correct_mallard, incorrect_mallard, correct_duckling, incorrect_duckling**: Lists of indices into the original query list

**duckling_regressions**: List of indices for queries that Duckling gets incorrect but Mallard gets correct

**duckling_missed_entities**: List of queries, expected sys_entities, actual sys_entities that Duckling completely
misses an entity for but Mallard identifies correctly 