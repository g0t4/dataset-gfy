btw you can compare the changes using:
`diff_two_commands 'cat  input-body.json | jq .messages.[-1].content -r' 'cat output.json | jq .content -r'`

I wanted to modify all my test cases to add a new assertion that I'd return another data point each... the before crossover index... so I asked gptoss to do this and it quickly made the changes!
- I have yet to validate all of its new numbers but I'll get there in a minute

Also note RAG grabbed the impl of these test cases! always useful to see though I suppose that could've confused the model w.r.t. it not yet actually returning these new numbers


Would've taken me 10 minutes to find and make the changes (let alone carefully validate my work)... I can skip the changes part and focus largely on validation and production code!
- Also noteworthy I had it rewrite the entire file... instead of selecting what to edit... I just passed the entire file for a rewrite and gptoss is so damn fast (200 TPS in this case)... it was NBD to rewrite all of it! Another huge win
