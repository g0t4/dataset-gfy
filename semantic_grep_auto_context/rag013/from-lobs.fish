#!/usr/bin/env fish

# * raw model output (1+ lines) - NOT json
cat logs | gsed -En '/Parsing input with format/,/Parsed message/ p' | gsed -En '$! p' \
    | string replace --regex '^\w\w\w \d\d \d\d:\d\d:\d\d \w+ llama-server\[\d+\]: ' '' \
    | string replace 'Parsing input with format GPT-OSS: ' '' >output-raw.harmony

# * parsed, response message object - JSON
# Dec 10 11:20:49 build21 llama-server[2671]: Parsed message: {"role":"a
cat logs | grep -E "llama-server\[\d+\]: Parsed message: {" \
    | string replace --regex "[^{]*" "" \
    | jq >output-parsed-message.json

# * rendered prompt .__verbose.prompt (after jinja template) - JSON
#   sync requests (non-streaming) also have .__verbose.content (model response)
#   this is the final SSE in streaming requests
#   TODO this is sometimes split across lines w/ no prefix on the line after... use gsed to find that (look for next line(s) without llama-server\[\d+\]: prefix?
# Dec 10 11:20:49 build21 llama-server[2671]: data stream, to_send: data: {"choices":[{"
cat logs | grep -E "llama-server\[\d+\]: data stream, to_send: data: {" \
    | string replace --regex "[^{]*" "" \
    | jq >input-rendered-prompt.json

# * body of original HTTP request (notably .messages)
cat logs | grep -E "llama-server\[\d+\]: srv  log_server_r: request:  {" \
    | string replace --regex "[^{]*" "" \
    | jq >input-messages.json
