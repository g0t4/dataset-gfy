Qwen is a refactoring powerhouse! fixed a ton of hardcoded constants and moved them to a new models.lua module
- 1785205657-trace.json => move hard coded model names to use local_share module constants that already exist
- 1785210455-trace.json => split out model constants from local_share module into new models.lua module
