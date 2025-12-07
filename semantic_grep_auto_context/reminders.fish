
# btw these are ordered same as llama-server's logs (last is first here)
#   make sure you have --verbose --verbose-prompt enabled to get all of the logs needed
pbpaste | string replace --regex "[^{]*" "" | jq > input_messages.json
pbpaste | string replace --regex "[^{]*" "" | jq > input-rendered-prompt.json
pbpaste | string replace --regex "[^{]*" "" | jq > output-parsed-message.json
pbpaste > output-raw.harmony # BTW you will need to trim log prefixes on each line (level/llama-server/time/etc)... I don't yet have a quick command to do that when capturing, I can set that up later... I need one like my pbsse that strips everything before { on a line

# reviewing:
cat input-rendered-prompt.json  | jq .__verbose.prompt -r | tree-sitter highlight --scope source.harmony
# uses my harmony TS grammar: https://github.com/g0t4/tree-sitter-harmony
# + highlights.scm queries: https://github.com/g0t4/dotfiles/tree/master/.config/nvim/queries/harmony
# + TS user config.json: https://github.com/g0t4/dotfiles/tree/master/.config/tree-sitter
#
# OR use your neovim config and open the harmony file(s)

