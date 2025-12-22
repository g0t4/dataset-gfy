Classic AI use case => adding missing imports!
This was after splitting a class out into its own file, without bringing any imports with.
Instead of copying everything from old file and then pruning.
Or, manually adding one at a time.
Just let AI FIM do it for me... literaly just stick cursor at top of file, into insert mode, done!

Case a): Now, it did struggle with project specific imports (it got third party ones and python modules right)...  b/c my RAG context didn't activate b/c the current line is empty AND there were no lines above so I want to do a case B) where I add one import and then on the next line I start more imports, that will trigger RAG and see what we get

Case b): I am adding one import first, then I will trigger a new FIM to see if the RAG matches get lucky and include imports needed for this FIM task!
 LOOK AT THAT! RAG MATCH 1 had a whole boat load of relevant imports!!! and it got all the imports right now (both local modules and 3rd party + python core)

Case c): doing some other work today and I ran into another import scenario... this time gptoss120b had to guess the import and got it right... when I changed  `ctags.read_file_lines` to `files.read_file_lines` and then moved cursor up to imports
