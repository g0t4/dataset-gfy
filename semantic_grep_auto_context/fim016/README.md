great FIM demonstration => reasoning is clearly covering whether to add global state to track current zoom level OR infer it... and then it went infer route

## RAG improvement notes

- I need to revisit overlapping RAG matches...
  - skip overlapping matches (i.e. function inside of a class in python... since each is chunked and often the class is matched as well as several of its methods)
  - take later matches (after k=5 cutoff or w/e threshold)
- ALSO, with gptoss I should have a higher threshold for # RAG matches? maybe 7 or maybe int(k*1.2)... some ratio of the typical num matches to include?
- AND I should allow RAG matches from current file that are OUTSIDE of the line range I am including in the FIM user message at the end...
  - in this case I strongly suspect my `on_key` function would match and that might lead to me not even needing to start the function definition to get it to complete the rest... like imports it would just see the missing function and do it!)
