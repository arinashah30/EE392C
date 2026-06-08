"""InferenceRunner for Qwen1.5-MoE / Qwen2-MoE on Neuron.

Direct clone of mixtral.mixtral_runner.MixtralRunner with Mixtral->QwenMoe
substitutions and the qwen_moe.* import path.
"""
import torch
from modules.config import MoENeuronConfig
from qwen_moe.neuron_modeling_qwen_moe import (
    NeuronQwenMoeForCausalLM,
    NeuronQwenMoeModel,
)
from runner import InferenceRunner
from transformers import AutoTokenizer

from neuronx_distributed.parallel_layers.checkpointing import _invoke_preshard_hook


class QwenMoeRunner(InferenceRunner):
    def load_hf_model(self):
        return NeuronQwenMoeForCausalLM.load_hf_model(self.model_path)

    def load_neuron_model_on_cpu(self, max_prompt_length, sequence_length, batch_size, **kwargs):
        # On CPU we can only run tensor parallelism with degree 1.
        hf_config = self.get_hf_config(sequence_length=sequence_length, **kwargs)
        neuron_config = self.get_config_for_nxd(
            hf_config,
            batch_size,
            1,
            max_prompt_length=max_prompt_length,
            sequence_length=sequence_length,
            enable_bucketing=False,
            **kwargs)
        hf_config.torch_dtype = torch.float32

        self.init_ditributed_env()
        neuron_model = NeuronQwenMoeModel(neuron_config)

        state_dict = NeuronQwenMoeForCausalLM.get_state_dict(self.model_path, neuron_config)

        _invoke_preshard_hook(neuron_model, state_dict)

        neuron_model.load_state_dict(state_dict, strict=False)

        if hf_config.torch_dtype == torch.bfloat16:
            neuron_model.bfloat16()

        model = NeuronQwenMoeForCausalLM(None, neuron_config)
        model.context_encoding_model.model = neuron_model
        model.token_generation_model.model = neuron_model
        return model

    def load_tokenizer(self, padding_side=None):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        # Qwen tokenizers expose `<|endoftext|>` as both eos and pad_token candidate.
        # Mirror the Mixtral fallback (use unk_token if pad_token missing) but degrade
        # gracefully if unk_token is also absent.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = (
                tokenizer.unk_token if tokenizer.unk_token is not None else tokenizer.eos_token
            )
        tokenizer.padding_side = padding_side if padding_side else self.get_padding_side()
        return tokenizer

    def get_config_cls(self):
        return MoENeuronConfig

    def get_model_cls(self):
        return NeuronQwenMoeForCausalLM

    def get_padding_side(self):
        return "right"

    def get_default_hf_generation_config_kwargs(self):
        config = super().get_default_hf_generation_config_kwargs()
        config['pad_token_id'] = 0
        return config


if __name__ == "__main__":
    QwenMoeRunner.cmd_execute()
