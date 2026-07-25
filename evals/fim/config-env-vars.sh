# Model under test (llama-server instance you're evaluating)
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=foo

# Judge model (a separate llama-server instance/model used to grade llm_judge cases)
# Only needed if any case being run uses grader:llm_judge. Point this at a model you
# trust to reason about code correctness -- doesn't need to be the model under test.
export JUDGE_BASE_URL=http://localhost:8081/v1
export JUDGE_API_KEY=foo
export JUDGE_MODEL=some-bigger-local-model
