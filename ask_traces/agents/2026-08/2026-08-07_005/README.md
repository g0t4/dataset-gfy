ds4 I wanted to add web chat viewer support for deepseek-v4-flash native FIM format prompts... I thought I had that for qwen native prompts (but I didnt lol)
- I only realized the qwen part after I told ds4 I had qwen native already working...
- ds4 got qwen and deepseek native prompts working for web!
  followed like what we have with gptoss chat completions FIMs... that shows diff at the top...
  man oh man ds4 is a fucking insanely good model

- only complaints...
  a few times ds4 was stopped (I just typed resume)... so NBD there... I wonder if I have a token limit on turns or if smth was going awry in my harness that caused it to stop
  TODO research stops if they happen again
  TODO also look into blocking my chat viewer from my FIM predictions code... as the model writes tokens FIMs are being triggered it looks like ... OR maybe the input event fires when it shouldn't
     the real solution to this tooling issue is to attach to buffers IF predictions should be working in them and not globally to all buffers like I have now... a good task for ds4 :)
