# Dynamic grouping reference fixtures

These synthetic tick states cover round-start history truncation and changing groups for 4, 6 and 10 players. The six-player physical slots are `[0, 1, 2, 5, 6, 7]`.

The `*.labels.npz` files were produced by an independent implementation of the same label-generation profile: pose-only relevance (`action=0`), EMA alpha 0.60, updates every 48 frames, threshold 0.30, coverage objective, sum capacity, and idle-hub recovery. All 24 arrays are checked, including scores and communication costs. They were generated with NumPy 2.4.4 and also match exactly under the project's NumPy 1.26.4.

The fixtures complement the five real bundled 4-player examples, whose membership happens to remain constant within each clip. No original dataset package is needed to run these regression tests.
