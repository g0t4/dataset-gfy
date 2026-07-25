# abbrs run from evals/fim -- expand into a full run_eval.py invocation
# (port = model under test, judge-port = grader). Named by port, not model
# name -- which model is actually on a given port on paxy.lan can change
# (systemd --user services get swapped in/out), so run_eval.py itself
# prints the real model name it got back from the response; trust that
# over the abbr name.
abbr eval_8010 "uv run --project .. python run_eval.py --port 8010 --judge-port 8013 --save"   # currently: glm47flash.service
abbr eval_8011 "uv run --project .. python run_eval.py --port 8011 --judge-port 8013 --save"   # currently: gemma-4-26b-a4b-it.service
abbr eval_8012 "uv run --project .. python run_eval.py --port 8012 --judge-port 8013 --save"   # currently: qwen3.6-35b-a3b-mtp-8bit.service
abbr eval_8013 "uv run --project .. python run_eval.py --port 8013 --judge-port 8012 --save"   # currently: gptoss120b.service (judge swapped to 8012 to avoid self-judging)
