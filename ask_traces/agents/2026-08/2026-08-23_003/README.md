deepseek-v4-flash-0731 uses apply_patch very well it seems! two uses here!
- I was working on cancel support for it last night and left it on and used deepseek today a few times... wow impressive
   TODO test how well deepseek does with more traces... vs how it does w/o apply_patch
   this has to be an improvement over rewriting entire files and/or rewriting parts with python scripts? or no? OR?
   TODO measure if smaller to use apply_patch or python code?
- stellar use of `py_compile` to validate python code changes!
  - deepseek did try to use `py_compile` standalone and that failed
    => then switched to `python -m py_compile` which worked
    and then thereafter used as module in subsequent calls
    so, deepseek notices failures and then avoids repeating them (at least in this small test case)
    TODO setup evals to see if a model repeats a mistake or always corrects course
      what would reliably fail and have a common workaround?

