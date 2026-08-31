# Network Architecture Scaling

This is a deferred extension of the core network architecture, not part of the
initial implementation.

## Locality-partitioned populations

A large hidden network could eventually be divided into many moderately sized
populations. Each population would use efficient local connectivity, while
cross-population connections would use sparse projections between designated
structural boundary groups:

```text
local population → sparse boundary projection → local population
```

Boundary or “edge” neurons are structural routing groups, not semantic
features. They must not be confused with feature populations used by the
interaction layer.

This could improve scaling when the runtime exploits block-local storage,
sparse communication, parallel population updates, or distribution across
devices. Splitting populations alone does not guarantee efficiency; population
size, scheduling overhead, boundary bandwidth, and cross-population timing
would need to be measured.

The existing population-level `ConnectionSpec` model is a suitable starting
point. A later extension could let a connection select structural source and
target groups without changing the logical view of the SNN as one network.
