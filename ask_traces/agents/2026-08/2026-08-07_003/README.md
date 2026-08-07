of course, deep seek can wire up its own FIM based chat completions messages and request params for controlling thinking and reasoning effort!
- two small issues (it didn't have a way to test so NBD)...
  - the pathing it used to fim_dev.md was wrong ( I fixed to use absolute path which is fine for now)
  - max_tokens cut off the test completion I did b/c it defaults to 200 which is not deepseek's mistake... though I guess it could've asked me to set that or how to set that...

I am just thrilled to see it work with my general design ideas and not needing to bang out the code myself
- that said I almost would've preferred reuse gptoss elements verbatim (not copy) but meh
