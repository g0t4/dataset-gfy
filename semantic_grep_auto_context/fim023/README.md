in this case I turned off RAG to avoid external influences on how I build this new SSE denormalizer

thus, straightforward, local file as context only! worked great...

once shortcoming, it had to guess about log:debug and in my case I don't have that defined, I use log:info mostly
- totally sane guess! but w/o RAG there's no way to know short of other auto context (i.e. yanks/ctags)
