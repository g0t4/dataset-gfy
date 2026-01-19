Seemingly basic example, that is anything but awesome!

This is a tricky case of FIM in the middle of a line with a prefix and suffix...
- very common for a model to correctly complete the middle but then repeat the full line (including prefix/suffix)
    - actually I could ask the model to do this and then employ a diff tool to find overlaps...
    - but presently I ask for careful completions and give a similar example of this in the prompt b/c this is a common theme of many FIM's... basically, finish the middle of this line for me!
- Note: prefix has `    return ` (4 spaces before `return`) and (1 space after `return`)
        suffix has `b` only
        thus, cursor has space before it and `b` after it
        therefore the completion is `a - `
        which results in `    return a - b`

Heuristic:
- Go ahead and finish the code, generate full lines... then subtract the existing prefix/suffix and that is what you need to return!
  Keep in mind, prefix is not just lines before, it's also the cursor line's prefix text before the cursor/FIM_MIDDLE marker.
  Same for suffix, it's not just lines after, it's the cursor line's suffix text after the cursor/FIM_MIDDLE marker.
