experiment with a new approach to evolving code... ask model to show me current behavior (i.e. __str__ on a dataclass)... then ask for change to __str__ and model should run test case again when done to show me the new __str__ output
   think of this as a temporary unit test that's manually verified for lifespan of chat thread... in fact it could easily be promoted to areal test when done with changes!

   HRM maybe have it write it like a pytest then? and run w/ assertions?! cool! TDD like?

   FYI, changes to: simple changes to CrossoverData.__str__ and RmsPoint.__str__

TODO => add prompts to support this pattern!
  => make clear I want simple python script to test and modify some code
  => ok to even write that to disk somewhere (i.e. unit test) or just keep in conversation
  => every time a change is made run applicable test script again so I can see impact
     i.e. for __str__ tweaking on dataclasses
