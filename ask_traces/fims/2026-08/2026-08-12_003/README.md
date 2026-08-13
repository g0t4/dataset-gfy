## Summary

Muse Glimmer testing of a very open ended FIM

- **reasoning off is the only one that gave me what I watned**
- **reasoning NEVER** gave me what I wanted
  AND, only one reasoning suggestion was even remotely useful/good IMO: (origin which is origin/HEAD and that matches oh too)

## subset of FIM:

```fish
# reset
abbr grhh 'git reset --hard HEAD' # last commit hard reset
#
abbr grh_undo_amend_commit 'git reset --hard HEAD@{1}' # use reset --hard to undo the last amend commit using reflog
# btw HEAD -> branch and it is the branch that is reset (not HEAD)... but use HEAD as a bookmark to mean current branch so you don't have to specify the branch

abbr grsh 'git reset --soft HEAD~1' # previous commit soft reset to review it and then purge by grhh or gco etc
abbr groh <|CURSOR_IS_HERE|>
# clean
#   (FYI leave --dry-run so I can remove it as last arg, then I don't need a second set of abbrs... b/c I should always do a quick review dryrun / or interactive)
abbr gclean 'git clean -d --dry-run' # --dry-ru[n], entire [d]irectories
abbr gcleani 'git clean -d --interactive' # [i]nteractive is alternative to dry-run
abbr gcleanx 'git clean -d -x --dry-run' # -x == ignored files too
abbr gpristine 'git reset --hard && git clean -dffx'
```
## What I had in mind
- `abbr groh` was the abbr I wanted (I typed that)
- I had `ORIG_HEAD` in mind `git reset --hard ORIG_HEAD`
  ** FYI backticks `` are merely delimiters here (not part of completion)

## reasoning_off/1786596546-trace.json returned (ignore trailing blank lines)
`abbr groh 'git reset --hard ORIG_HEAD'`

- good: exactly what I wanted!
  fast b/c non thinking!
- complaints: duplciated cursorline prefix (NBD as my FIM tool strips that)
- notes:
    - did this w/ my hack to disable thinking (not in official jinja template, not yet)
      1. set reasoning_strength="off" (not officially supported but IMO gives a hint to not think)
      2. force empty thinking by setting assistant message at end
          {
            "content": "",
            "reasoning_content": "<|eom|><|start|>assistant to=user<|message|>",
            "role": "assistant"
          },

## reasoning_low/1786596556-trace.json:
`abbr groh 'git reset --hard HEAD~1'`

- complaints: duplicated cursorline prefix
  terrible suggestion IMO, I literally already have that as `grhh` a few lines above!

## reasoning_low/1786596567-trace.json
`'git reset --hard origin'`

- good: didn't duplicate the cursorline prefix
- good suggestion: IIUC `origin` => `origin/HEAD`
  I haven't ever thought to use this but ok... maybe
  but I don't know how often I'd use this, to reset to origin's HEAD (vs maybe @{upstream} which seems more plausible)

## reasoning_medium/1786598494-trace.json
`'git reset --hard origin'`

-- see low notes for same FIM



## reasoning_high/1786598522-trace.json
`'git reset --hard origin/$(git_current_branch)'`

- good: no dupliated cursorline prefix


## reasoning_high/1786598606-trace.json and reasoning_high/1786598687-trace.json
`'git reset --hard HEAD~1'`

- terrible suggestion
- but at least no duplicated cursorline prefix like with low reasoning above when it suggested the same thing


