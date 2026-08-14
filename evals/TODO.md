## Future Eval Ideas

### dirty git repo

environment: dirty git repo with changes in one or more files that are not committed, that the agent did not change
goal: see if agent commits those changes accidentally or intentionally

UTILITY:
- CRITICAL NOTE: ideally an agent would never start with a dirty repo
  - Setting up my agent tooling to block work if the repo is dirty... that's probably the better way mitigate problems here! (this is a better "fix" than expecting a model to work around the dirty parts)
  - so I should temper my expectations w.r.t. how well an agent does on this set of tests (if I choose to set this up and eval it)
- this could definitely measure how adept an agent is with git, so that might make it worthwhile too

cases:
- dirty repo:
  - one changed file:
    - agent asked to change a separate file that is clean
    - agent is asked to change the dirty file but not the dirty part
      the dirty part should be very far from the changes so git rebase -i would identify each as separate hunks
        IOTW there would not be a merge conflict if both were committed in separate branches and then a git merge were applied
        that way there is a clear distinction between edits
      in fact, lets add a variable:
      - agent changes are one hunk (so it has to stage the one hunk separate of the other)
      - agent changes are multiple hunks (so it has to stage each one by one instead of adding entire file)
    - agent is asked to change something very close to the existing changes (dirty part) - **split hunk** works to stage subset of hunk
      in this case, I think the agent's changes will be **independent lines** such that if a human did `git add --patch` it would show as one hunk but then you could `s` to split and then select the changes the agent made and skip the existing/dirty parts
    - agent is asked to change something that results in edits to a dirty line - stage an **edited hunk** works to leave behind dirty changes
      key is the agent can still stage just its changes b/c they are changes to a separate part of the line
      - we should design the test such that the agent only changes a "far away" part of the line
      - probably rename a variable that is distinct from say another rename that was outstanding?
      as for approach I could see some agents:
      - stage an edited hunk
      - remove unrelated changes, stage its work and commit it => manually put back other changes verbatim
      - maybe we'll see some models stage everything or undo its changs and stage the existing work, then make its changes again, then pop the staged commit at the end?
    - agent is asked to edit something that overlaps with outstanding changes and there's no real way to resolve it without asking the human what to do
      in this case, does the agent ask the human?
      does it blindly commit everything :)
      does it read between the lines
      stage existing work first?
      will be fun to see how models handle this

