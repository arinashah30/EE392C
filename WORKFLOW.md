ENABLE_DGE_NOTIFS=1 ./profiler/capture_and_export.sh  --model "Llama-3.2-1B-Instruct"   --tp 4 --lnc 1   --batch 1 --ctx 128 --seq 256

cp /dev/shm/traced_model/Llama-3.2-1B-Instruct-nxdi-lnc1-tp4-b1-ctx128-seq256/profile/i-0be799b9ee00c90da_pid_2180468_nc_*_model_446048307616134.json /dev/shm/new_llama_trace/
cp /dev/shm/traced_model/Llama-3.2-1B-Instruct-nxdi-lnc1-tp4-b1-ctx128-seq256/profile/profile.json /dev/shm/new_llama_trace/

python3 -m dmsim.cli ingest   --profile-dir /dev/shm/new_llama_trace  --model-key 446048307616134   --min-transfer-bytes 1 --output data/traces/llama32_decode_4core_min1_no_unknown.json --skip-unattributed-dma

python profiler/visualize_trace.py data/traces/llama32_1b_decode_4core_min1_no_unknown.json

python3 -m dmsim.cli compare   --trace data/traces/llama32_1b_decode_4core_min1_no_unknown.json   --baseline-hierarchy configs/hierarchy/trainium2_baseline.yaml   --candidate-hierarchy configs/hierarchy/trainium2_diff_mem_50sbuf_25hbm.yaml   --baseline-policy configs/policies/baseline_hbm.yaml   --candidate-policy configs/policies/decode_tiered.yaml   --output data/traces/sim_results_dma_lit.json
