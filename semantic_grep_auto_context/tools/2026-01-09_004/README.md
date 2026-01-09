repo level refactorings w/ tool use!!!!

refactor combine two properties (AVTime+float) into a single RmsPoint dataclass
- update CrossoverData which used min/max (AVTime) and min_rms_value/max_rms_value (float)...
  merge into just min/max (RmsPoint)... then followup to update consumers of the CrossoverData type

such a joy to think about WHAT design I want and not waste my time on HOW to refactor

btw... gptoss was not tripped up by gaps.py semantic_grep matches
- even though gaps.py uses identically named min_rms_value/max_rms_value variables that are not part of CrossoverData
