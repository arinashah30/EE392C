
export HF_TOKEN=""   # your token

# 0) Expand /dev/shm (default is ~half RAM; profiling needs more)
sudo mount -o remount,size=192G /dev/shm

# 1) Project venv for dmsim (missing on this clone — recreate it)
cd /home/ubuntu/EE392C
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2) Weights download — use Neuron venv (already has huggingface-cli)
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export HF_HOME=/dev/shm/huggingface
export HUGGINGFACE_HUB_CACHE=/dev/shm/huggingface/hub
export TMPDIR=/dev/shm/tmp
mkdir -p /dev/shm/huggingface/hub /dev/shm/tmp

huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir /dev/shm/Llama-3.2-1B-Instruct \
  --token "$HF_TOKEN"

huggingface-cli download Qwen/Qwen1.5-MoE-A2.7B \
  --local-dir /dev/shm/Qwen1.5-MoE-A2.7B