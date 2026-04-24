# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import warnings

import torch

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_BEFT_TARGET_MODULES_MAPPING,
)

from .layer import BEFTLayer, Linear


class BEFTModel(BaseTuner):
    """
    Creates a Infused Adapter by only fine-tuning the added bias terms of value projections from a pretrained
    transformers model in low-training-data regimes (BEFT). The method is described in detail in
    https://arxiv.org/abs/2509.15974

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`BEFTConfig`]): The configuration of the (BEFT) model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The (BEFT) model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from peft import BEFTModel, BEFTConfig

        >>> config = BEFTConfig(
        ...     peft_type="BEFT",
        ...     task_type="SEQ_2_SEQ_LM",
        ...     target_modules=["v"],
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> beft_model = BEFTModel(model, config, adapter_name="default")
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`BEFTConfig`]): The configuration of the (BEFT) model.
    """

    prefix: str = "beft_"
    tuner_layer_cls = BEFTLayer

    @staticmethod
    def _create_new_module(beft_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = beft_config.fan_in_fan_out = False
            new_module = Linear(target, adapter_name, **kwargs)
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear` is supported."
            )
        return new_module

    def _create_and_replace(
        self,
        beft_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        kwargs = {
            "fan_in_fan_out": beft_config.fan_in_fan_out,
            "init_beft_weights": beft_config.init_beft_weights,
        }

        if isinstance(target, BEFTLayer):
            target.update_layer(
                adapter_name,
                beft_config.init_beft_weights,
            )
        else:
            new_module = self._create_new_module(beft_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_BEFT_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_BEFT_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config
