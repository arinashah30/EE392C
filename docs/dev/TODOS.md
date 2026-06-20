# To-Do

- ~~Update the access latency logic/incorporate interconnect latency~~
- ~Update~/verify the energy computation and add refresh for StRAM and HBM
- Remove "unknown" handling
- Verify the tensor placement logic, consider best-case worst case placement
  - ~Add fallback/eviction-to configuration~
- Verify the tensor mapping heuristics — see [TENSOR_MAPPER_AUDIT.md](TENSOR_MAPPER_AUDIT.md) (June 2026; NEFF slot shapes vs HF, viewing/utilization, P0 mapper bugs)
- ~~Verify access latency logic~~
- ~~remove on-chip to on-chip traffic~~
- Change eviction to LRU
- Bring back cross-domain traffic
- deepest_enabled currently by [-1], should be end of graph or defined
- Note: wipe_levels_on_boundary in hierarchy defines wiped kernels

