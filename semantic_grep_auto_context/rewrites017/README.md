why do step by step refactors!

First, (in dir `a-strip-np-array`) have the model strip np.array() around arrays in test cases (both input and output values for 3 test cases)
Second, in dir `b-inline` have the model inline those same variables - six inlines across the three tests

Now, imagine if you had 30 tests like this to refactor!

Still is a challenge to make sure the model doesn't materially alter the values of the array items, or otherwise change the test cases...
but, that often can be handled by creative use of icdiff (and similar tools)
hence the multi-step refactor here, across to AskRewrite requests, making it easier to verify in two parts
- I tend to break down refactor requests such that a diff is easier to achieve b/c my time validating that is critical to minimize
