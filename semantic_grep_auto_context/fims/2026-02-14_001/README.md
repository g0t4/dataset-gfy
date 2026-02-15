## 1771134327-thread.json /  1771134337-thread.json
when moving code (extract function), let the AI tell you what params are needed for the extracted function!
- even does type hints (especially when the prompt pushes for these in a given language)

- BTW it is easy to force the model to generate non existant params (i.e. lines_to_remove) => 1771134337-thread.json
  - the first model response was all I needed but by adding a `,` I forced it to guess about another param, arguably it should've returned nothing...
    so be wary of forcing it too hard!
    of course later on when getting it to pass params it suggests passing one for the not needed second param... b/c the param is listed! valid to suggest

## 1771135030-thread.json
when moving code (extract function), just let the model generate the needed imports (no need to move them all and then prune)... just go into insert mode!

## 1771135806-thread.json
then to use the new function go to the consumers and just let the AI generate the params to pass (RAG FTW!)
- btw one param was bogus but that's cuz I had a bogus param added to the extracted function: `lines_to_remove`

