deepseek is a boss with run_xonsh (and without run_process to use instead!)

deepseek added a new macro global function for iterm2 for my existing applescript based "open/refresh inspector" that I had in Keyboard Maestro
- now I can have deepseek port it to lua via AXUIElement model!
- and I can fix the bug in the applescript to open the inspector if not open already (iterm2 changed some button text AFAICT)

tiny gripe:
- deepseek tried to use `python3 -c` and pass a command to run a python script
  instead of just running the python with xonsh directly
  that said, when it failed, deepseek ran the python directly in xonsh so NBD it actually understands that you can use xonsh to run python code too!
