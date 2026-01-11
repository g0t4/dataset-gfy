## recall-github-org-thread.json

I needed the github org for the gitsigns.nvim plugin, in my nvim lazy config:

```lua
    {
        "<CURSOR>gitsigns.nvim"
    },
```

and it filled in the org:

```lua
    {
        "lewis6991/gitsigns.nvim"
    },
```

BTW... it is VERY important to verify what it fills in is legit... in the context of repos, packages, etc... always double check
- neat thing is, if the model generates it (from its "memory")... then you go find it and check the spelling matches...
  this can be better than just copy/paste alone
  both you searching and the model's weights together serve as a "double check"

Also, this can be useful to avoid typing out long repo names.

## recall-org-and-repo-plus-config-thread.json

example of recalling both: org/repo

plus in this case it gave me a config example
