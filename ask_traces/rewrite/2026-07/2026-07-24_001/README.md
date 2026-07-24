notes: first GLM rewrite trace! not bad at all

good:
  - does exactly what I wanted (sans a minor bug)
  - and it left my comments alone!
  - correctly indexed at 1 for first character in `sub(1,`

bad:
  - sub(1, 11) is needed because VIRTUAL_ENV is 11 chars long
    I wonder if it forgot the `_` counts too!
    OR, if a base0/1 issue where end was set with base0 in mind
    - this might be forgiveable especially because neovim mixes base0/base1 parameters all the time
    maybe GLM has a problem "counting" :)
