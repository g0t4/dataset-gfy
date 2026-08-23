deepseek added special behavior for fd_command when in a git repo...
- keep in mind it couldn't see callers to this function so it had no idea that I had file vs dir pickers
  this would need adaptation (possibly) for files/dir/both/etc but in the git context probably only files ever make sense
- also issue with empty ref "git cat-file -p :path<shift+alt+f>" which implies HEAD IIUC

this is not the end all be all solution as there are so darn many alterations of this but it is a GREAT start point!
 - and proopsing to use `git ls-files` is fantastic too (instead of `fd`)
