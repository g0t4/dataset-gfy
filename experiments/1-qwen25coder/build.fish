#!/usr/bin/env fish

cat gfy.jsonl | grep -v '^\s*//' | jq

mkdir -p out
cat gfy.jsonl | grep -v '^\s*//' > out/gfy.nocomments.jsonl

