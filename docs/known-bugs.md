# Known Bugs

This document describes known bugs and problems that hinder our progress.
After discovery, the descriptions will be updated until closed.

## You spin me round

- **Date Identified:** March 30, 2026
- **Status:** Open
- **Issue nr**: #27
- **Description:**\
  The brittle star agent exhibits a continuous spinning motion instead of navigating toward the target.
  We suspect this behaviour stems from the environment's reset logic.
  Upon `env.reset()` (triggered by truncation or termination), the target is relocated to a random position along a fixed circular radius.
- **Root Cause:**\
  The agent is likely over-optimizing for the reset transition.
  Because the target’s distribution is constrained to a circle, a high-frequency spinning policy ensures the agent is consistently oriented toward or sweeping through the target's next potential spawn point.
  This suggests the reward function or the reset frequency may be incentivizing "positional hedging" over active pursuit.
- **Potential Fixes:**
  - **Target Randomization:**\
    Vary the radius of the target spawn during reset, rather than keeping it on a fixed circle.
  - **Initial State Noise:**\
    Introduce random initial joint angles for the brittle star at the start of each episode to prevent it from locking into a circular motion pattern.
  - **Penalty Adjustment:**\
    Review the penalty for truncation to ensure the agent doesn't find more value in ending the episode quickly to "re-roll" the target position.
