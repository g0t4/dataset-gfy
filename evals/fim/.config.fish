# abbrs run from evals/fim -- expand into a full run_eval.py invocation
# (port = model under test, judge-port = grader). Named by port, not model
# name -- which model is actually on a given port on paxy.lan can change
# (systemd --user services get swapped in/out), so run_eval.py itself
# prints the real model name it got back from the response; trust that
# over the abbr name.
abbr eval_8010 "fim_evals --port 8010 --judge-port 8013 --save" # currently: glm47flash.service
abbr eval_8011 "fim_evals --port 8011 --judge-port 8013 --save" # currently: gemma-4-26b-a4b-it.service
abbr eval_8012 "fim_evals --port 8012 --judge-port 8013 --save" # currently: qwen3.6-35b-a3b-mtp-8bit.service
abbr eval_8013 "fim_evals --port 8013 --judge-port 8012 --save" # currently: gptoss120b.service (judge swapped to 8012 to avoid self-judging)

function fim_evals
    set project $DATASETS_REPO/evals
    set script $DATASETS_REPO/evals/fim/run_eval.py
    uv run --project $project python3 $script $argv
end

if not test -f ~/.config/fish/completions/fim_evals.fish
    echo generating fim_evals completions
    register-python-argcomplete --shell fish fim_evals > ~/.config/fish/completions/fim_evals.fish
end
