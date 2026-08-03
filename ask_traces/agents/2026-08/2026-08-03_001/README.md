OVERALL VERY IMPRESSED WITH deepseek-v4-flash-0731! I might have to get myself another 6000 Pro RTX :)... for my other models

first task given to deepseek-v4-flash
- when editing files => it didn't just rewrite the full file, it passed `old` and `new` strings to a python edit script
  - this definitely saves some time/tokens on smaller changes
  - hopefully on bigger changes it splits things up into smaller edits across the file or just rewrites if touches most of the file
  - I suspect this model will do very well with an Edit Predictions tool I've been thinking about... if it can think about old/new code chunks then that's exactly what I'd want for my own Edits Predictions (give me old lines and new lines)
  - I wonder if the model is ever trained to select the old code by some keyword that's unique... like find line with "foobar5" and replace next 20 lines after it (or until next unique blurb)
    this would cut down on tokens in old string which might be good, might not be!
- using ripgrep as requested (and not ls/grep)
  - multi tool calling! helps for code searching/reading
- understands logic well, faithfully moved rag_cancel/rag_request_ids without taking on any other changes (kept original code as close as was possible for the task, didn't get distracted with "nearby" changes)
- followed instructions!
  - committed religiously (as requested in system prompt)
     - even used the author name I asked for (static with Qwen3.6...) ... maybe that is my next task, to make that dynmamic :) based on current model at time of trace starting
  - used `luac` to check for simple syntax errors
  - reviewed `git diff` before committing (did this twice, but didn't do it on middle change)
  - added type annotations for rag_request_ids and rag_cancel to Prediction type (again, as requested in system prompt)
- understood using object equality as a replacement for the request_ids check (not that it is on a prediction object that we can compare whereas before the code was on rag_request_ids matching)
- amended commit author correctly once we were done, fixed the qwen part to deepseek and I didn't even give it the full author to use, I had it redo it by replacing qwen part which is in name and email
  - correctly amended last 3 commits to change author and then it checked its work with the git log in a compact format with author name on front of each line!

monitor these things going forward:
- I have to assess these more over time... but a few things to watch for:
  - used `git add -A` several times, two of which it did use a `git diff` right before, so those are ok
    but the middle time it should've done a diff before and/or after
    TODO prompt models to check what is staged before committing? or at least do this when you `git add -A`?
-

complaints:
- left behind a logic bug in comparing stale RAG responses (after a new RAG request sent, if an older request responds we don't want that to trigger FIM on old results)...
  - when I pointed it out, it correctly fixed it and immediatley it used the word `stale`... so it knew exactly what part I was talking about as I never mentioned stale in my follow up
  - this is a minor issue... best long term path is to setup tests to catch this instead of relying a strict reading and understanding of the code
