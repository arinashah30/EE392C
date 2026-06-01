ENABLE_DGE_NOTIFS=1 ./profiler/capture_and_export.sh  --model "Llama-3.2-1B-Instruct"   --tp 4 --lnc 1   --batch 1 --ctx 128 --seq 256

cp /dev/shm/traced_model/Llama-3.2-1B-Instruct-nxdi-lnc1-tp4-b1-ctx128-seq256/profile/i-0be799b9ee00c90da_pid_2180468_nc_*_model_446048307616134.json /dev/shm/new_llama_trace/
cp /dev/shm/traced_model/Llama-3.2-1B-Instruct-nxdi-lnc1-tp4-b1-ctx128-seq256/profile/profile.json /dev/shm/new_llama_trace/

python3 -m dmsim.cli ingest   --profile-dir /dev/shm/new_llama_trace   --model-key 446048307616134   --min-transfer-bytes 1   --skip-unattributed-dma   --no-aggregate-dma   --max-access-events 0   --output data/traces/llama32_1b_decode_4core_min1_no_unknown.json

python profiler/visualize_trace.py data/traces/llama32_1b_decode_4core_min1_no_unknown.json

python3 -m dmsim.cli compare   --trace data/traces/llama32_1b_decode_4core_min1_no_unknown.json   --baseline-hierarchy configs/hierarchy/trainium2_baseline.yaml   --candidate-hierarchy configs/hierarchy/trainium2_diff_mem_50sbuf_25hbm.yaml   --baseline-policy configs/policies/baseline_hbm.yaml   --candidate-policy configs/policies/decode_tiered.yaml   --output data/traces/sim_results_dma_lit.json

python3 -m dmsim.cli compare   --trace data/traces/llama32_1b_decode_4core_min1_no_unknown.json   --baseline-hierarchy configs/hierarchy/trainium2_baseline.yaml   --candidate-hierarchy configs/hierarchy/trainium2_diff_mem_25hbm.yaml   --baseline-policy configs/policies/baseline_hbm.yaml   --candidate-policy configs/policies/decode_ltram_only.yaml   --output data/traces/sim_results_dma_lit.json

python3 -m dmsim.cli compare   --trace data/traces/llama32_1b_decode_4core_min1_no_unknown.json   --baseline-hierarchy configs/hierarchy/trainium2_baseline.yaml   --candidate-hierarchy configs/hierarchy/trainium2_diff_mem_50sbuf_25hbm.yaml   --baseline-policy configs/policies/baseline_hbm.yaml   --candidate-policy configs/policies/decode_tiered.yaml   --output data/traces/sim_results_dma_lit.json
=== iso-area budget (constant die area) ===
  hbm_capacity_after: 77309411328
  hbm_nominal: 103079215104
  hbm_removed_bytes: 25769803776
  ltram_area_um2: 187416754734.5454
  ltram_capacity_bytes: 4685418868363
  ltram_replaces_hbm_fraction: 0.25
  sbuf_capacity_per_core_after: 14680064
  sbuf_nominal_per_core: 29360128
  sbuf_removed_per_core_bytes: 14680064
  stram_area_um2: 48131357.3770
  stram_capacity_bytes: 66180616
  stram_replaces_sbuf_fraction: 0.5
  stram_scope: per_core
  sbuf_capacity_bytes: 14680064
  stram_capacity_bytes: 66180616
  ltram_capacity_bytes: 4685418868363
  hbm_capacity_bytes: 77309411328

=== trainium2_baseline / baseline_hbm ===
workload: chip_cores_0_1_2_3
total_time_ns: 4,446,607  (worst core nc0)
total_energy_pJ: 1,670,092,155,884
refresh_energy_pJ: 1,666,616,161,824
hbm_read_bytes: 23,416,580
hbm_write_bytes: 2,763,056
hbm_traffic_bytes: 26,179,636
kernel_wipes: 5736

transfers_by_hop:
  hbm->sbuf: 4070
  sbuf->hbm: 134396

time_by_core_ns:
  nc0: 4,446,607
  nc1: 4,440,986
  nc2: 4,444,042
  nc3: 4,446,353

energy_by_level_pJ:
  hbm: 1,667,365,479,771
  sbuf: 2,726,676,100

=== trainium2_diff_mem_50sbuf_25hbm / decode_tiered ===
workload: chip_cores_0_1_2_3
total_time_ns: 4,413,934  (worst core nc0)
total_energy_pJ: 140,827,660,844,847
refresh_energy_pJ: 140,821,014,202,912
hbm_read_bytes: 15,970,428
hbm_write_bytes: 2,763,056
hbm_traffic_bytes: 18,733,484
kernel_wipes: 5736

transfers_by_hop:
  hbm->sbuf: 2774
  ltram->sbuf: 868
  sbuf->hbm: 134396

time_by_core_ns:
  nc0: 4,413,934
  nc1: 4,408,670
  nc2: 4,411,815
  nc3: 4,412,940

energy_by_level_pJ:
  hbm: 826,826,090,812
  ltram: 9,532,600
  sbuf: 2,339,561,921
  stram: 139,998,485,659,648

=== comparison (candidate vs baseline) ===
{
  "baseline": "trainium2_baseline",
  "candidate": "trainium2_diff_mem_50sbuf_25hbm",
  "time_ns": {
    "baseline": 4446607.0260902075,
    "candidate": 4413934.2668554755,
    "pct_change": -0.7347795531070433
  },
  "energy_pJ": {
    "baseline": 1670092155883.6238,
    "candidate": 140827660844846.94,
    "pct_change": 8332.328739987219
  },
  "hbm_traffic_bytes": {
    "baseline": 26179636,
    "candidate": 18733484,
    "pct_change": -28.442534495132016
  }
}