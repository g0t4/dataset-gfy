FYI for reviewing... I would mostly look at the last user message + output.json's content
```bash
# last user message:
cat 1767508751/input-body.json | jq .messages[-1].content -r
# this is my prompt + selected code when running :AskRewrite command

# generated replacement text
cat 1767508751/output.json | jq .content -r

```


FYI after testing, so far I haven't stumbled on any bugs either! So, a big win.

Most of these requests revolve around creating XLimits and YLimits "DTOs" to wrap get_xlim/get_ylim (matplotlib) with named properties (start/end, bottom/top)... plus computed properties for width/height, middle, half (of width/height)

These save me the time of making the changes myself, instead I can focus on verification which is important even if I do the changes... so why not let the model do the bulk of the conversion and I just review it!

Note I broke this down into many smaller requests, I could've done one huge (tool use) request to have the model do it all... but a few things:
- if that huge conversion fails and/or doesn't go the way I like, I just wasted review time
- ALSO, along the way I was deciding what I wanted to do for next steps, i.e. stumbling on YLimits and realizing that would also benefit from this new wrapper

Long term I would like to get comfortable tasking the model with larger refactorings. I just have to get a feel for how that works best w.r.t. my taste, reviews and also my tooling for this

NOTE: in many requests, just by including the following in the prompt, RAG pulled the relevant code for inclusion!
    XLimits
    get_x_limits()

ALSO yanks often had relevant information for generation
