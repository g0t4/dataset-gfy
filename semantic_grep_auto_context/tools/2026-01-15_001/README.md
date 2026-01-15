wanted to look up a keymap in my fish shell config...

asked gptoss in agent mode (w/ tools)
+ semantic_grep tools

it found what I wanted right away!

this is fun!

I am going to flesh out this workflow and maybe add some special features for it...
.. the ability to talk to my codebases, FRIGGIN AWESOME

on my own machine too

lightning fast

## complaint
only complaint, it had the answer in my auto context RAG matches! but, ran semantic_grep again and basically got the same results...

in fact, it googled (semantic_grep) for the name of the function:
```
- semantic_grep:
    {
      "query": "fzf_nested_file_unrestricted_widget bind",
      "top_k": 20,
      "filetype": "fish"
    }
```

which is literally right next to the keymap's key combo... but TBF, the keycombo was in a diff RAG match that it clearly missed:
```
bind_both_modes_default_and_insert alt-shift-u _fzf_nested_file_unrestricted_widget
```

- TODO/MAYBE should I put the user request before the auto RAG matches again? I used to have it that way... not sure if that increases odds here or not... also NBD really b/c this was still so darn fast!

