I just added this alias for is_upward() called is_local_min()

gptoss reasonably suggested is_minimum() ... but semantic_grep matches didn't include is_local_min() so it didn't suggest that!
- NBD just a good example of a RAG failure! (RAG can't be perfect... never expect it to be)
- And in this case, an impetus to consider revamping the FIM RAG query (think prompt) to include some other aspect/context to maybe match this in the future...
   ... i.e. in this case, if I included the datatype of the variable cross0. that I was completing... then that would lead to really good RAG matches
   - also maybe a better example of adding a new type of context (not RAG specific) that grabs relevant symbols from an LS... IOTW grab LSP client's completions! and let the model pick from that too
