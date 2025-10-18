## Experiment ideas

- Vary amount of non-swear responses to see how it affects sweariness overall
- Measure sweariness before and after
- Measure perplexity before and after (how confident is the model in the old vs new style?)
- Train variants of my LoRA adapter and swap/blend them...
    - and do this at runtime w/o reloading base model

## Separate fine tune datasets

- Emoji heavy vs light vs healthy levels vs none
    - Modify dataset per model tendencies, i.e. qwen3 is emoji heavy whereas qwen2.5-coder is not

## Observations

- swear words, especially humorfully placed (i.e. "brackets and bullshit" is god level)
- occasionally emotional starts, i.e. _rolls eyes_ are good but not always!
- dry humor YES
- humor, YES! DRY, YES PLEASE!

## TODOs

- Should I worry about not having longer examples?
    - i.e. explaining a concept in depth, writing code, etc?

## Past prompts

> typeof: typeof NaN === 'number', seriously?

> what are some aspects of javascript that are humorous, and/or would work well with swear words when explained
