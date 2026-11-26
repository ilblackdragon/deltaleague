# Conditional follow-up using an independent GRPO direction

The seed-3 checkpoint fdfd46a4 scores 59.38 on public MATH500 versus 48.44 for d0120ae6. Its BBH score is lower (37.04 versus 42.59). Instead of copying whole modules, compare ways to introduce the difference between the seed-3 model and its public GRPO parent eff1e1b7. Apply each rule to decoder projection matrices in layers 0 through 22.

- The first candidate fills coordinates where the leader still equals eff1e1b7 but the seed-3 model differs.
- The second adds half of the seed-3 update to the leader in FP32, then rounds once.
- The third restores the common ancestor value only where the leader and seed-3 updates have opposite signs.

Use the same two paired 18-benchmark panels, then a newly selected 94-question panel that excludes the first routing round. [Conditional protocol](protocol.json)
