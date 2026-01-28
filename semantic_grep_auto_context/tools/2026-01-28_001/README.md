I am impressed with how darn good gptoss120b is... this was good work!
- saved me at least 3 minutes of shotgun surgery!
- it did a stellar job updating the detectors list consumer... even though I wanted that nuked!
    I didn't realize it was essential to keep due to the return early case (missed in cursory review)
    so I should've asked it to remove that too, instead I just followed up! worked well!

Dings:
- failed to delete original classes (gave up?)... it tried, for sure
   - in first attempt, it appears to have added an extra empty line
   - attempt2 was to add context before?! and keep two lines b/w functions
     HRMMM makes me think, it is really used to python code with two lines b/w functions
     maybe I should just set that setting on in yapf (IIRC I set it to one b/w top level funcs)
     ... OR can I instruct in the prompt:
     > "when using apply_patch, careful with blank lines b/w functions... most people use 2 but I use 1?)"

     OR smarter apply_patch would try to find a close fit and point this out!
        histogram alone should've found like 99% match and then it could say hey did you add an extra \n?

     ALSO consider:
     > "don't forget you only need to include enough UNIQUE context to match, you should avoid going beyond that b/c you might make a mistake!"
     >    MINIMAL CONTEXT to be unique!

     AND:
     > "if you repeatedly have trouble applying an edit, try breaking it down into smaller edits"
     >   "i.e. around functions, how about edit ONE at a time"!

- I need to get it to not leave comments behind when removing stuff
    > "this is a git repo, NEVER leave comments about removed code... ALWAYS get rid of old code!"
- why did it edit my class GapContext just to change its description?!
    I assume it is b/c it says it needs to change classes to functions in that file (but I never asked to change that class!)
    and it didn't change it! thankfully... but it changed the description lol
    and my god to do that it rewrote (verbatim) the entire class lol when it could've done one line to replace b/c the description line was definitely unique!
       again, I assume it did this for the patch b/c it was thinking in terms of replacing the entire class
       and the habit of changing the description kicked in

