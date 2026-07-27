GLM-4.7-Flash traces adding virtual text for FIM prediction reasoning and then changing how its styled
- overall this model does seem highly capable as an agent!\
- *** reminds me of Qwen ALOT! like very similar steps, double checks and then fixes albeit it did not do a final check
- 1785141647-trace.json - add reasoning as virtual text (rest of lines after thinking dots)
   - one big complaint, the code works but GLM changed my split_lines and broke it!
      BUT it appears this is just an issue with how it changed the file and not the intent of the model
      it does seem to misunderstand the original file (likely due to json serialization issues.. yup.. at one point it mentions the original file had \\r\\n which is wrong...
         TODO one problem with my tooling... when I show the file contents in the chat viewers ... I split on \n which yikes too... I think I have a bug in my chat viewer so I should look into that! (not GLMs fault for that part and the chat viewer doesn't likely affect GLM however I should dig into if there is a bug in how I report back to the model and make sure I didn't mess it up!!
         commit was `git show 39ecb989`
      OR just not rewrite the entire file! sheesh!
   - escaped the escapes which resulted in splitting lines in weird places (n/r letters IIRC)
   - otherwise mostly ok code
   - it also changed some comments that weren't related (non-destructive changes) but still WTH!
- 1785141975-trace.json - make thinking dots green
