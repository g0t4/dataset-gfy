converting tests to use new API, similar but totally different assertions, this is a huge time savings! for just one test!
- note I had copied an example test (already ported) from same file to clipboard (yanks)
- and then RAG found other relevant matches too! other test cases as references
    in fact, retrieval only returned other tests using the new APIs!

PERFECT!
## second thread
port rest of tests
- love how well it did with exit code being zero now when previously I didn't return it if zero
- also previously I would check STDERR was EMPTY, whereas new API doesn't see empty STDERR/STDOUT
