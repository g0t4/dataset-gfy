## hump dir

real code => again, curve ball having it find the spot to edit instead of me selecting it... many advantages, including it can edit any files then (not just what I select with AskRewrite)
- TODO I need to habituate selecting text and then AskQuestion /tools too... like I would with AskRewrite... except the model uses tool calls to edit anything!

FYI I do not know that this solved the problem I had, but it was a good example of tool use so I included this before I've had a chance to get back to the scenario and see what I think of its solution

## formatter dir

tool use changes to my new apply_patch formatter, I didn't want to make the change myself... so why not give a curve ball to tool use! (not point to exactly what needs changed, let it do that)
- first time was good but it renamed things and stripped off comments
- so I followed up a few times to get it to knock off most of that
- last request though, it still changed at least one variable name
  TODO refine prompt to get it to stop this... probably need developer message to sternly demand not to do this
- FYI it ran a ton of semantic_grep calls that overlapped, NBD... I didn't look to see why but it must've not found what it wanted OR it was hunting for other usages?
