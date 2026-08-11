# GBSC source compatibility

This directory migrates the executable UCI path in
`algorithm/python/GBSC-main/GranularBallUCI.py` and
`GranularBallUCISC.py`.

- Granular balls use source weighted-density division with minimum size 8,
  then fixed-radius normalization with factor 2.
- The affinity is the source signed boundary distance divided by `sqrt(d)`,
  squared inside the Gaussian kernel; overlapping distances are not clamped.
- The executable source uses `sigma=1.0`; the paper UCI parameter table lists
  `sigma=0.1`. The Benchmark follows the executable source by default.
- Source loading Min-Max normalization is handled centrally by Benchmark;
  granular balls retain sample indices so labels can be mapped correctly.
