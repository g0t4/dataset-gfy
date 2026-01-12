tried a minimal prompt, selected some code and hoped RAG would do the rest :)... looks like gptoss also used sed a bit

yup, RAG found the CrossoverData type for me, no need to spell that out!

neat to see gptoss hunting, carefully (not too many lines) for context to be able to update the original code to use the new function in plot_helpers...
- and also making sure there aren't other spots using this code

btw for some reason I don't see the change for the caller code... not sure if patch file format is correct... might've ended patch before second update...
- wonder if I should encourage gptoss to double check its work more often? always make sure edits are applied, don't trust the tool worked
