## 1770144928-thread.json
prompt gptoss to gen tests for complex code I wanna change (add coverage + new, failing tests)...

review/tweak tests (human)
=> start new chat thread (scratch)

BTW final model apply that failed, I ran myself instead of having it redo it... I need to look into failure, might've been max_tokens...

## 1770149301-thread.json

*new thread (clean slate, no holdover from test generation above)*

prompt gptoss to run tests
=> reports back 2 failing

prompt gptoss to fix them (by adding new feature)
=> edits files
=> runs tests to verify

DONE!

**Failing Tests** were the only description!

BTW, part of why I did this, *I wasn't sure how I wanted to solve this new constraint*, so I thought **why not ask the model to come up with a solution* and see how I feel about it... maybe even run generation a few times and pick best solution!?

## caveats

tests were not fully baked, I had left out one condition for min/max constraint
actual datatype CrossoverData is not hashable so equality fails in that case
