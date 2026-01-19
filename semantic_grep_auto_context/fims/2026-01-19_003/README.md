## 1st take 1768856343-thread.json
I just added my new reminders/warnings about the cursor line prefix/suffix...
... and drum roll... gptoss is killing it on low reasoning level!!!

FYI terrible formatting of reminders, a MESS! lol and still it helped
- I didn't add the heading... really all that was probably doing it is warning when the prefix of cursor line is all whitespace!


## 2nd take 1768856619-thread.json
Fixed up the reminder/warnings... and it's still doing well (one time it had messed up extra tab but that was 1 in 15 at most)
- PRN maybe ONLY mention when it's all whitespace ahead of it?
- then again this was helping with whitespace in the FIM middle of line too `return <CURSOR>b' => 'a - ' was reliably coming up too! so I wanna keep all reminders for now
    if anything put a caveat right before to explain why they're included

## 3rd take 1768856923-thread.json
Fixed reminders to have heading and added new line to split heading from first reminder
- gptoss responds well to this too!

FYI one concern is that it might not see the other full context FIM spot now that I added these reminders, time will tell how well this does in other contexts? cross fingers!

My intutition was if I can point out these tricky examples when they arise instead of just try to cover them in examples... that this would make the difference to make it obvious when to pay attention to my caveat examples!
- Heck might even be worth NOT showing examples UNLESS a specific situation occurs that matches it?
