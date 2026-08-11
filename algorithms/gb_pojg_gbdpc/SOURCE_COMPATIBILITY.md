# GB-POJG-GBDPC source compatibility

This directory migrates the existing Python reproduction in
`algorithm/python/GB-POJG/gb_pojg` and follows the released MATLAB source:

- `GenerationMethodGBs.GBPOJG` provides the granular-ball binary tree,
  pruning, and anomaly split;
- `ClusteringMethod.GBDPC` provides density, delta distance, decision-value
  center selection, and label propagation;
- `gamma=0` and `delta=0.5` are the GBDPC defaults in official `main.m`.

The paper pseudocode differs from the released source in three details. The
Benchmark implementation intentionally preserves source behavior:

1. split threshold: `max(delta * sqrt(n), n ** 0.25)`;
2. root `BestValue`: `-1`;
3. anomaly test `MeanNumAll`: sum of selected-ball sizes, not their mean.

The algorithm receives only `X`; Benchmark preprocessing, true labels, and
metrics remain outside this directory.
