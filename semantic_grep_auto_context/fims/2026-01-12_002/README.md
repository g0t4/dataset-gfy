love fixing imports w/ RAG + FIM w/ gptoss

NOTE in this case RAG did not find the import, largely because I didn't even start it to give a hint that it would be relevant.
- the only signal would be far away in the same file (the recently added safely....
- if I included recent edits, that would be a good way to get better RAG matches in this case
  - PRN maybe if the line has no prefix, also include recent edits? nothing fancy, just take from neovim's list of `:changes` ... for current file, wow that would actually be super useful
- TODO should I add in the file name of the FIM file at the top of the user message showing it?
