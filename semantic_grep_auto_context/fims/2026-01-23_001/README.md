two FIMs for same code (one letter diff `_`)

first was (12) for 12th button (as the comments below point out is doable)
then I add `_` making `button_` and now it's matching button_by_description and yanking out the description which is also below in comments!

this is why I love FIM+RAG... where RAG query works much like LSP completions (for current line's prefix text...)
- except with LLM you can complete the intimate context too! not just one symbol too.. but the rest of the line/block/etc!
   i.e. the "Shape Style" description in comments below!

Also, in this case, RAG matched usages of an API and not the API itself... interesting!
- still works great!
