## part1 - hide toolbar in matplotlib plot

once again, no need to point where, just ask what I want and the model finds the spot...
- and this is matplotlib so it's not mission critical if this is not the "right" spot or "right" way to do this...

## part2 - replace usages of toolbar's message text with a text box

demonstrates - building upon previous changes

btw it looks like my thread export logic, after a first request/response, is duplicating the response message into the thread as well as into the response message (formerely "output.json")... I will have to fix that later
part 2 - ask to replace toolbar message use w/ a text box
- note gptoss made an "Implementation steps" (checklist)
- clever use of top_k=1 (1 result) for `on_motion_notify_event(` and it got that back!
- several failures along the way, with context for patches... yet it kept trying and did check what was going on... I didn't check what caused them... only know that it produced working code when I reviewed changes.
