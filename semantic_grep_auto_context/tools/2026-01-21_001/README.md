move new AskRewrite completion function and make it available in AskQuestion too

## follow up - other prompts.lua

my instructions were vague about prompts.lua
- initially, it put the `SlashCommandCompletion` function in `lua/ask-openai/frontends/prompts.lua`
  - which honestly makes more sense organizationally... given the other two command files are literally named `frontend.lua`!
- BUT, it should've seen that that one doesn't have M.slash_commands...

So, I asked it to move it and it got it right!
> "ok cool itt looks good! but you added to wrong prompts.lua module!"
- I wasn't specific, I didn't tell it where the right one was, it found it anyways
- gptoss used chat history (past tool calls) to locate the other one, no new tool calls needed!
    - often I've noticed it seems to do a good job not querying for the same info
    - NOT always, but plenty of times
    - especially when making subsequent changes (follow up changes), it mostly jumps to apply_patch tool calls
- gptoss is good about editing its own code, that it just wrote too!

## detected my changes when moving

I had added logging for my testing and left it in!
- at first when it tried to remove SlashCommandCompletion from the wrong prompts.lua... it failed b/c I had changes there
- gptoss immediately went to dump the contents and see and thus found my changes
    log:info("iSHIT")
    log:info(vim.inspect(result))
  - also at the top, an import
- Yes, I have penchant for logging w/ swear words, it's super easy to search and get unique hits in logs :)
    (unless I leave them in, lol)

## misc
- love that it accurately figures out paths for imports!
  - seems like when gptoss knows what needs to be done, it will get it done
  - only issues I ever see are when it isn't paying attention to tasks that need to be done
  - thankfully I don't see it give up on tasks it knows to do
     - I only see it behave as if some steps were not clear
  - makes me think, having a plan step might be very beneficial...
     - add a "plan" tool, force its use (maybe /plan or /force_plan)?
     - that would capture the plan
     - have args to modify plan / add steps to plan
         - this would be where encouraging (in prompt) to add to plan, I think would result in it finding missed steps that otherwise might be ignored
     - could even have /followup_plan to have an auto message at end (after assistant says its done) to then ask it to double check versus its plan (send plan back with this)
         - could also let me as a user follow up with that in float window too (if I forgot, or wasn't sure if I wanted it initially)...
     - I wonder how this might help (or not)
  - also, another idea, prompt separately for the plan while it is working on request (parallel prompt) and then have that ready for when it alleges to be done
     - could even run several prompts w/ diff temperatures or instructions
        - and have a merge step where its asked to consolidate and cleanup into one list

