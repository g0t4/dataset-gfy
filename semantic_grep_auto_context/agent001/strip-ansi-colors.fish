#!/usr/bin/env fish

gsed -r "s/\x1B\[[0-9;]*[[:alpha:]]//g"
