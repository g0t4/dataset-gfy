 this is exactly the test case I needed, and interestingly the test name alone was not necessarily that helpful to understand expectations
- and yet... it set the hops to land EXACTLY on a frame (frame 5 past the start of the gap... hence 35th overall frame
- correctly set expectation that before/after would be the frame itself!!
   - this could've easily been taken to mean 34/36 ... if not really undersatnding the purpose of the frame numbers

INCREDIBLE TO SEE ITS MATH REASONING:
> So need min at exact integer frame number => frame_num_before == frame_num_after. That's when min seconds * FPS is integer. FPS=30. seconds = 1.0 + h/882. So (1.0 + h/882)*30 integer => 30 + (h*30)/882 = 30 + h/29.4 integer => h/29.4 integer => h multiple of 29.4 => h = 29.4 * k. Need integer h. For k=5, h=147 which is integer? 29.4*5=147. Yes h=147 hops.
> ...
> frame = (1.0+147/882)*30 = (1+0.1666667)*30 = 35? 1+147/882 = 1+0.1666667 =1.1666667 sec *30 =35 exactly.

these are the calcualtions I would've done to derive this test!

I am impressed with gptoss120b's math abilities
