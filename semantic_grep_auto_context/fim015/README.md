I literally just created the module with clip_gap_only()... RAG found it when FIM'ing in the click handler of my matplotlib module... and gptoss120b (via FIM) suggested exactly what I wanted!

Nothing like seeing how fast RAG can provide brand new code, in a far away place (diff file in a big codebase)
- First three RAG matches were exactly what I needed
- This affirms that when the FIM line is blank, it is useful to take a few lines before for RAG purposes!
  Literally this is what triggered the matches:

```
  # save mp4 clip
  <CURSOR_HERE>
```
