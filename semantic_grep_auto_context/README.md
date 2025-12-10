## Capture types

`rewrites` - captures from my AskRewrite command in neovim where I select some code, ask for a change and then it provides auto context (via RAG, yanks, etc) and then I get back new code to replace the selected code ... I also have other context I can selectively choose to add like /file but you won't see that b/c it is stripped out by the command and used to include the full file, you will see the full file in the prompt instead!
`yanks` - mostly FIM requests that relied on yanks to know what to insert, could be AskRewrite too - see prompt to discern
`rag` - mostly FIM requests that relied on RAG for crucial context, though could be AskRewrite too - see prompt to discern
