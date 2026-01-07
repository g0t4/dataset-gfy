btw compare the changes:
```
diff_two_commands 'jq .request_body.messages[-1].content -r *-thread.json' 'jq .response_message.content -r *-thread.json'
```

FYI on a small scale, this is like what I enjoy about AskQuestion /tools... no need to specify WHERE to make the change, just WHAT to change!
- though, in this case, limited to a single file
- maybe it would be more efficient to just have it use tools to do this instead? and add a AskRewriteFile? just an idea

love select all in a file, then rewrite => specify change and hit return... wait a few seconds, and done!
- in this case a small change, to pretty print JSON files...
- big win is that it can add imports at top of file too!
- gptoss faithfully reproduced the entire file, with the one change requested!
  - I did have to redo and tell it not to touch my comments

*** it is incredible that I give no guidance about rewriting the full file, and yet it does it!
- it wouldn't surprise me for it to be like, rewrite these lines only and make a markdown guide... and yet it doesn't!
   and I've done this type of request a few times now!

TODO add a /rewrite_file slash command to simplify this?
- no need to select all...
- perhaps optimize/customize the prompt for this purpose...
  - tell it to leave my unrelated comments ALONE! (put into dev message?)
