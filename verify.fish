#!/usr/bin/env fish

cat test.jsonl | grep -v '^\s*//' | jq
