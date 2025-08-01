# coding=utf-8
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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
import copy
import math
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, HybridCache
from ...configuration_utils import PretrainedConfig, layer_type_validation
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_rope_utils import rope_config_validation
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, is_torchdynamo_compiling, logging
from ..auto import AutoModel
from ..gemma2.configuration_gemma2 import Gemma2Config
from ..gemma2.modeling_gemma2 import (
    Gemma2MLP,
    Gemma2PreTrainedModel,
    Gemma2RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from ..gemma3.modeling_gemma3 import (
    Gemma3Attention,
    Gemma3DecoderLayer,
    Gemma3ForCausalLM,
    Gemma3RMSNorm,
    Gemma3TextModel,
    Gemma3TextScaledWordEmbedding,
)
from ..paligemma.modeling_paligemma import (
    PaliGemmaCausalLMOutputWithPast,
    PaliGemmaForConditionalGeneration,
    PaliGemmaModel,
    PaligemmaModelOutputWithPast,
)
from ..timm_wrapper.configuration_timm_wrapper import TimmWrapperConfig


logger = logging.get_logger(__name__)


class Gemma3nTextConfig(Gemma2Config, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Gemma3nTextModel`]. It is used to instantiate an
    Gemma3nTextModel model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Gemma 3n E4B, e.g.
    [google/gemma-3n-E4B](https://huggingface.co/google/gemma-3n-E4B).

    Configuration objects that inherit from [`Gemma3nTextConfig`] and can be used to control the model outputs. Read
    the documentation from [`Gemma3nTextConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 262400):
            Vocabulary size of the Gemma3nText model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`Gemma3nTextModel`]
        vocab_size_per_layer_input (`int`, *optional*, defaults to 262144):
            Vocabulary size of the per-layer text embeddings that augment the standard embeddings.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        hidden_size_per_layer_input (`int`, *optional*, defaults to 256):
            Dimension of the hidden representations for per-layer emebeddings.
        intermediate_size (`int` or `Sequence[int]`, *optional*, defaults to 16384):
            Dimension of the MLP representations. MatFormer configurations may wish to provide a sequence of integers
            to account for vairable intermediate_size values across layers. In such cases,
            `len(intermediate_size) == num_hidden_layers`.
        num_hidden_layers (`int`, *optional*, defaults to 35):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout this
            [paper](https://arxiv.org/pdf/2305.13245.pdf). If not specified, will default to `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 256):
            The attention head dimension.
        hidden_activation (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the decoder. Will default to
            `"gelu_pytorch_tanh"` if not specified. `"gelu_pytorch_tanh"` uses an approximation of the `"gelu"`
            activation function.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings used in gloabl attention.
            NOTE: if you apply new rope type and you expect the model to work on longer `max_position_embeddings`, we
            recommend you to update this value accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        rope_local_base_freq (float, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings for local attention.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        sliding_window (`int`, *optional*, defaults to 512):
            This is the size of the sliding window used by local attention layers.
        layer_types (`Optional`, *optional*):
            A sequence of strings defining the attention type for that layer as either "sliding_attention" or
            "full_attention". If not provided, `layer_types` will de inferred from `num_hidden_layers` using a pattern
            of four "sliding_attention" layers followed one "full_attention". The last layer in the model should always
            be a "full_attention" layer.
        final_logit_softcapping (`float`, *optional*, defaults to 30.0):
            Scaling factor when applying tanh softcapping on the logits.
        altup_active_idx (`int`, *optional*, defaults to 0):
            The index of the prediction from which AltUp will compute additional predictions or correct
        altup_coef_clip (`float`, *optional*, defaults to 120.0):
            The maximum amplitude of an AltUp prediction or correction coeficient weight.
        altup_correct_scale (`bool`, *optional*, defaults to `True`):
            If True, apply the `AltUp.correct_output_scale` to the corrected prediction at `altup_active_idx`.
        altup_num_inputs (`int`, *optional*, defaults to 4):
            The number of predictions that AltUp should be make given the input sequence.
        num_kv_shared_layers (`int`, *optional*, defaults to 15):
            The number of layer that share KV cache values. During the forward pass, the last `num_kv_shared_layers`
            layers in the model "share" the KV values in that each local and global layer in this range uses the KV
            cache values computed for the last local or global layer, respectively, before entering this range. The
            value should be `num_kv_shared_layers` should be a scalar of `sliding_window_pattern`.
        laurel_rank (int, *optional*, defaults to 64):
            The intermediate size for the linear projections in the Learned Augmented Residual Layer.
        activation_sparsity_pattern (Sequence[float], *optional*, defaults to `(0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)`):
            The sparsity factor used to extract the top-k activations for a given layer. The provided Sequence must
            explicitly provide a sparsity value for each layer in the model.

    ```python
    >>> from transformers import Gemma3nTextModel, Gemma3nTextConfig

    >>> # Initializing a Gemma3nText gemma3n_text-E4B style configuration
    >>> configuration = Gemma3nTextConfig()

    >>> # Initializing a model from the gemma3n_text-E4B style configuration
    >>> model = Gemma3nTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "gemma3n_text"

    def __init__(
        self,
        vocab_size: int = 262_400,
        vocab_size_per_layer_input: int = 262_144,
        hidden_size: int = 2048,
        hidden_size_per_layer_input: int = 256,
        intermediate_size: Union[int, Sequence[int]] = 16_384,
        num_hidden_layers: int = 35,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        hidden_activation: str = "gelu_pytorch_tanh",
        max_position_embeddings: int = 32_768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        bos_token_id: int = 2,
        rope_theta: float = 1_000_000.0,
        rope_scaling: Optional[dict[str, Any]] = None,
        rope_local_base_freq: float = 10_000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        sliding_window: int = 512,
        layer_types: Optional[Sequence[str]] = None,
        final_logit_softcapping: float = 30.0,
        altup_active_idx: int = 0,
        altup_coef_clip: float = 120.0,
        altup_correct_scale: bool = True,
        altup_num_inputs: int = 4,
        num_kv_shared_layers: int = 15,
        laurel_rank: int = 64,
        activation_sparsity_pattern: Optional[Union[float, Sequence[float]]] = (0.95,) * 10 + (0.0,) * 25,
        **kwargs,
    ):
        PretrainedConfig.__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        if isinstance(intermediate_size, Sequence) and (intsize_len := len(intermediate_size)) != num_hidden_layers:
            raise ValueError(
                "intermediate_size must have an explicit intermediate size for every layer or one for all layers. "
                f"Expected {num_hidden_layers} values but got {intsize_len}."
            )
        elif not isinstance(intermediate_size, Sequence):
            intermediate_size = [intermediate_size] * num_hidden_layers

        self.vocab_size = vocab_size
        self.vocab_size_per_layer_input = vocab_size_per_layer_input
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.layer_types = layer_types

        self.rope_local_base_freq = rope_local_base_freq
        self.rope_scaling = rope_scaling
        rope_config_validation(self)

        if layer_types is None:
            self.layer_types = [
                "full_attention" if i % 5 == 0 else "sliding_attention" for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = layer_types

        layer_type_validation(self.layer_types)

        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.num_kv_shared_layers = num_kv_shared_layers

        self.altup_active_idx = altup_active_idx
        self.altup_coef_clip = altup_coef_clip
        self.altup_correct_scale = altup_correct_scale
        self.altup_num_inputs = altup_num_inputs

        self.laurel_rank = laurel_rank

        if activation_sparsity_pattern is None:
            activation_sparsity_pattern = [0.0] * num_hidden_layers

        if (len_asp := len(activation_sparsity_pattern)) != num_hidden_layers:
            raise ValueError(
                "activation_sparsity_pattern must have an explicit activation sparsity value for every layer."
                f"Expected {num_hidden_layers} values but got {len_asp}."
            )
        self.activation_sparsity_pattern = activation_sparsity_pattern


class Gemma3nAudioConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Gemma3nAudioEncoder`]. It is used to instantiate
    an `Gemma3nAudioEncoder` model according to the specified arguments, defining the model architecture. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the Gemma 3n E4B, e.g.,
    [google/gemma-3n-E4B](https://huggingface.co/google/gemma-3n-E4B).

    Configuration objects that inherit from [`Gemma3nAudioConfig`] and can be used to control the model outputs. Read
    the documentation from [`Gemma3nAudioConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 128):
            Vocabulary size of the additional hard-token embeddings for audio model. These augment the embeddings
            included in the `Gemma3nTextModel` to provide, e.g., the end of audio and audio soft token placeholder
            tokens when converting `input_ids` to embeddings in the `Gemma3nForConditionalGeneration` model.
        vocab_offset (`int`, *optional*, defaults to 262272):
            Offset between the tokenizer vocab index for the token ids embedded by `Gemma3nMultimodalEmbedder` and the
            0-indexed `Gemma3nMultimodalEmbedder.embedding` table.
        input_feat_size (`int`, *optional*, defaults to 128):
            The number of channels in each mel-spectrogram frame.
        hidden_size (`int`, *optional*, defaults to 1536):
            Dimension of the hidden representations.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        gradient_clipping (`float`, *optional*, defaults to 10000000000.0):
            Clipping value used to stablize extremely large gradient values.
        conf_attention_chunk_size (`int`, *optional*, defaults to 12):
            The sub-sequence size for local attention processing inside the Conformer ("conf") section of the
            Universal Speech Model.
        conf_attention_context_left (`int`, *optional*, defaults to 13):
            The left context size of the local attention inside the Conformer ("conf") section of the
            Universal Speech Model.
        conf_attention_context_right (`int`, *optional*, defaults to 0):
            The right context size of the local attention inside the Conformer ("conf") section of the
            Universal Speech Model.
        conf_attention_logit_cap (`float`, *optional*, defaults to 50.0):
            Logit cap applied during local attention inside the Conformer ("conf") section of the
            Universal Speech Model.
        conf_num_attention_heads (`int`, *optional*, defaults to 8):
            The number of attention heads in local attention inside the Conformer ("conf") section of the
            Universal Speech Model.
        conf_num_hidden_layers (`int`, *optional*, defaults to 12):
            The number of layers that use local attention inside the Conformer ("conf") section of the
            Universal Speech Model.
        conf_conv_kernel_size (`int`, *optional*, defaults to 5):
            Convolution kernel size for the conformer block inside the Conformer ("conf") section of the
            Universal Speech Model.
        conf_reduction_factor (`int`, *optional*, defaults to 4):
            Reduction factor used in the conformer block inside the Conformer ("conf") section of the
            Universal Speech Model.
        conf_residual_weight (`float`, *optional*, defaults to 0.5):
            Residual connection weight inside the Conformer ("conf") section of the
            Universal Speech Model.
        sscp_conv_channel_size (`tuple(int, int)`, *optional*, defaults to `(128, 32)`):
            The channel sizes for the first and second convolutional layers in the Sub-sample Convolution Projection
            ("sscp") section of the Universal Speech Model.
        sscp_conv_group_norm_eps (`float`, *optional*, defaults to 0.001):
            Epsilon used in group normalization in the subsample convolution projection in the Sub-sample Convolution
            Projection ("sscp") section of the Universal Speech Model.
        sscp_conv_kernel_size (`tuple(tuple(int, int), tuple(int, int))`, *optional*, defaults to `((3, 3), (3, 3))`):
            Kernel sizes of the two convolutional layers in the subsample convolution projection  in the Sub-sample
            Convolution Projection ("sscp") section of the Universal Speech Model. The kernel sizes are specified as a
            tuple of height and width for each layer, where the height corresponds to the time dimension and the width
            corresponds to the frequency dimension.
        sscp_conv_stride_size (`tuple(tuple(int, int), tuple(int, int))`, *optional*, defaults to `((2, 2), (2, 2))`):
            Stride sizes of the two convolutional layers in the subsample convolution projection in the Sub-sample
            Convolution Projection ("sscp") section of the Universal Speech Model. The stride sizes are specified as a
            tuple of height and width for each layer, where the height corresponds to the time dimension and the width
            corresponds to the frequency dimension.

    Example:

    ```python
    >>> from transformers import Gemma3nAudioConfig, Gemma3nAudioEncoder

    >>> # Initializing a Gemma3nAudioEncoder gemma3n_audio-E4B-style configuration
    >>> configuration = Gemma3nAudioConfig()

    >>> # Initializing a model from the gemma3n_audio-E4B style configuration
    >>> model = Gemma3nAudioEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "gemma3n_audio"

    def __init__(
        self,
        vocab_size: int = 128,
        vocab_offset: int = 262_144 + 128,  # text vocab size + vision vocab size
        input_feat_size: int = 128,
        hidden_size: int = 1536,
        rms_norm_eps: float = 1e-6,
        gradient_clipping: float = 10_000_000_000.0,
        conf_attention_chunk_size: int = 12,
        conf_attention_context_left: int = 13,
        conf_attention_context_right: int = 0,
        conf_attention_logit_cap: float = 50.0,
        conf_num_attention_heads: int = 8,
        conf_num_hidden_layers: int = 12,
        conf_conv_kernel_size: int = 5,
        conf_reduction_factor: int = 4,
        conf_residual_weight: float = 0.5,
        sscp_conv_channel_size: tuple[int, int] = (128, 32),
        sscp_conv_group_norm_eps: float = 1e-3,
        sscp_conv_kernel_size: tuple[tuple[int, int], tuple[int, int]] = (
            (3, 3),
            (3, 3),
        ),
        sscp_conv_stride_size: tuple[tuple[int, int], tuple[int, int]] = (
            (2, 2),
            (2, 2),
        ),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_feat_size = input_feat_size
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps
        self.vocab_size = vocab_size
        self.vocab_offset = vocab_offset
        self.gradient_clipping = gradient_clipping
        self.conf_attention_chunk_size = conf_attention_chunk_size
        self.conf_attention_context_left = conf_attention_context_left
        self.conf_attention_context_right = conf_attention_context_right
        self.conf_attention_logit_cap = conf_attention_logit_cap
        self.conf_num_attention_heads = conf_num_attention_heads
        self.conf_num_hidden_layers = conf_num_hidden_layers
        self.conf_conv_kernel_size = conf_conv_kernel_size
        self.conf_reduction_factor = conf_reduction_factor
        self.conf_residual_weight = conf_residual_weight
        self.sscp_conv_channel_size = sscp_conv_channel_size
        self.sscp_conv_group_norm_eps = sscp_conv_group_norm_eps
        self.sscp_conv_kernel_size = sscp_conv_kernel_size
        self.sscp_conv_stride_size = sscp_conv_stride_size


class Gemma3nVisionConfig(TimmWrapperConfig):
    r"""
    This is the configuration class to store the configuration for a timm backbone [`TimmWrapper`]. It is used to
    instantiate an timm model model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Gemma 3n E4B
    vision tower, e.g. [google/gemma-3n-E4B](https://huggingface.co/google/gemma-3n-E4B).

    Configuration objects inherit from [`Gemma3nVisionConfig`] and can be used to control the model outputs. Read the
    documentation from [`Gemma3nVisionConfig`] for more information.

    Config loads imagenet label descriptions and stores them in `id2label` attribute, `label2id` attribute for default
    imagenet models is set to `None` due to occlusions in the label descriptions.

    Args:
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        do_pooling (`bool`, *optional*, defaults to `False`):
            Whether to do pooling for the last_hidden_state in `TimmWrapper` or not.
        architecture (`str`, *optional*, defaults to `"mobilenetv5_300m_enc"`):
            Determines vision architecture for TimmWrapper.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        vocab_size (`int`, *optional*, defaults to 128):
            Vocabulary size of the additional hard-token embeddings for vision model.
        vocab_offset (`int`, *optional*, defaults to 262144):
            Offset between the tokenizer vocab index for the token ids embedded by `Gemma3nMultimodalEmbedder` and the
            0-indexed `Gemma3nMultimodalEmbedder.embedding` table.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.

    Example:
    ```python
    >>> from transformers import Gemma3nVisionConfig, TimmWrapper

    >>> # Initializing a TimmWrapper gemma3n_vision-E4B-style configuration
    >>> configuration = Gemma3nVisionConfig()

    >>> # Initializing a gemma3n_vision-E4B-style TimmWrapper from the configuration
    >>> model = TimmWrapper(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "gemma3n_vision"

    def __init__(
        self,
        initializer_range: float = 0.02,
        do_pooling: bool = False,
        architecture: str = "mobilenetv5_300m_enc",
        hidden_size: int = 2048,
        vocab_size: int = 128,
        vocab_offset: int = 262_144,
        rms_norm_eps: float = 1e-06,
        model_args: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.architecture = architecture
        self.initializer_range = initializer_range
        self.do_pooling = do_pooling
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.vocab_offset = vocab_offset
        self.rms_norm_eps = rms_norm_eps


class Gemma3nConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Gemma3nForConditionalGeneration`]. It is used to
    instantiate a Gemma3nForConditionalGeneration according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    Gemma3n-E4B.

    e.g. [google/gemma-3n-E4B](https://huggingface.co/google/gemma-3n-E4B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`Union[Gemma3nTextConfig, dict]`, *optional*):
            The config object of the text backbone.
        vision_config (`Union[AutoConfig, dict]`,  *optional*):
            Custom vision config or dict.
        audio_config (`Union[AutoConfig, dict]`,  *optional*):
            Custom audio config or dict.
        audio_soft_tokens_per_image (`int`, *optional*, defaults to 188):
            The number of soft tokens per audio clip.
        vision_soft_tokens_per_image (`int`, *optional*, defaults to 256):
            The number of soft tokens per image.
        boi_token_id (`int`, *optional*, defaults to 255999):
            The begin-of-image token index to wrap the image prompt.
        eoi_token_id (`int`, *optional*, defaults to 262144):
            The end-of-image token index to wrap the image prompt.
        image_token_id (`int`, *optional*, defaults to 262145):
            The image token index to encode the image prompt.
        boa_token_id (`int`, *optional*, defaults to 256000):
            The begin-of-audio token index to wrap the audio prompt.
        eoa_token_id (`int`, *optional*, defaults to 262272):
            The end-of-audio token index to wrap the audio prompt.
        audio_token_id (`int`, *optional*, defaults to 262273):
            The audio token index to encode the audio prompt.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.


    Example:

    ```python
    >>> from transformers import Gemma3nForConditionalGeneration, Gemma3nConfig, Gemma3nTextConfig

    >>> # Initializing a MobileNet vision config, which is loaded from TIMM
    >>> vision_config = Gemma3nVisionConfig()

    >>> # Initializing a Gemma3n Audio config
    >>> audio_config = Gemma3nAudioConfig()

    >>> # Initializing a Gemma3n Text config
    >>> text_config = Gemma3nTextConfig()

    >>> # Initializing a Gemma3n gemma-3-4b style configuration
    >>> configuration = Gemma3nConfig(text_config, vision_config, audio_config)

    >>> # Initializing a model from the gemma-3-4b style configuration
    >>> model = Gemma3nTextConfig(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gemma3n"
    sub_configs = {
        "text_config": Gemma3nTextConfig,
        "vision_config": Gemma3nVisionConfig,
        "audio_config": Gemma3nAudioConfig,
    }

    def __init__(
        self,
        text_config: Optional[Union[Gemma3nTextConfig, dict[str, Any]]] = None,
        vision_config: Optional[Union[Gemma3nVisionConfig, dict[str, Any]]] = None,
        audio_config: Optional[Union[Gemma3nAudioConfig, dict[str, Any]]] = None,
        audio_soft_tokens_per_image: int = 188,
        vision_soft_tokens_per_image: int = 256,
        boi_token_id: int = 255_999,
        eoi_token_id: int = 262_144,
        image_token_id: int = 262_145,
        boa_token_id: int = 256_000,
        eoa_token_id: int = 262_272,
        audio_token_id: int = 262_273,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(text_config, dict):
            text_config = Gemma3nTextConfig(**text_config)
        elif text_config is None:
            text_config = Gemma3nTextConfig()
            logger.info("text_config is None. Using default Gemma3nTextConfig.")

        if isinstance(vision_config, dict):
            vision_config = Gemma3nVisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = Gemma3nVisionConfig()
            logger.info("vision_config is None. Using default Gemma3nVisionConfig.")

        if isinstance(audio_config, dict):
            audio_config = Gemma3nAudioConfig(**audio_config)
        elif audio_config is None:
            audio_config = Gemma3nAudioConfig()
            logger.info("audio_config is None. Using default Gemma3nAudioConfig.")

        self.text_config = text_config
        self.vision_config = vision_config
        self.audio_config = audio_config

        self.audio_soft_tokens_per_image = audio_soft_tokens_per_image
        self.vision_soft_tokens_per_image = vision_soft_tokens_per_image
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.image_token_id = image_token_id
        self.boa_token_id = boa_token_id
        self.eoa_token_id = eoa_token_id
        self.audio_token_id = audio_token_id
        self.initializer_range = initializer_range


class Gemma3nModelOutputWithPast(PaligemmaModelOutputWithPast):
    r"""
    past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    audio_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        audio_hidden_states of the model produced by the audio encoder and after projecting the last hidden state.
    """

    audio_hidden_states: Optional[torch.FloatTensor] = None


class Gemma3nCausalLMOutputWithPast(PaliGemmaCausalLMOutputWithPast):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.text_config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder after projecting last hidden state.
    audio_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        audio_hidden_states of the model produced by the audio encoder and after projecting the last hidden state.
    """

    audio_hidden_states: Optional[torch.FloatTensor] = None


class Gemma3nRMSNorm(Gemma3RMSNorm):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__(dim, eps=eps)
        del self.weight
        self.with_scale = with_scale

        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer("weight", torch.tensor(1.0), persistent=False)

    def _norm(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = self._norm(x.float()) * self.weight.float()
        return output.type_as(x)


# ==== Audio Encoder ====


class Gemma3nAudioRelativePositionEmbedding(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig):
        super().__init__()
        self.config = config

        self.num_heads = self.config.conf_num_attention_heads
        self.channels = self.config.hidden_size
        self.head_dim = self.channels // self.num_heads
        self.max_backward = max(0, self.config.conf_attention_context_left - 1)
        self.max_forward = self.config.conf_attention_context_right

        self.pos_proj = nn.Linear(self.channels, self.num_heads * self.head_dim, bias=False)

        min_timescale = 1.0
        max_timescale = 1.0e4
        num_timescales = self.channels // 2
        log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / max(num_timescales - 1, 1)
        inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales) * -log_timescale_increment)
        self.register_buffer(
            "inv_timescales",
            inv_timescales.float().unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

    def _get_timing_signal_1d_pos(self, position: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        position = position.float().unsqueeze(-1)
        scaled_time = position * self.inv_timescales.to(device=position.device, dtype=torch.float32)
        timing_signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        return timing_signal.type(dtype)

    def _relative_shift(
        self,
        term_bd_before_shift: torch.Tensor,
        batch_size: int,
        num_heads: int,
        num_query_blocks: int,
        query_block_size: int,
        key_context_size: int,
        max_span_plus_1: int,
    ) -> torch.Tensor:
        """Performs the relative shift.

        Args:
          term_bd_before_shift: Tensor of shape [B, N, U, W, F_span]. batch_size
            (B), num_heads (N), num_query_blocks (U), query_block_size (W),
            key_context_size (C = W+L+R), max_span_plus_1 (F_span = L+R+1).

        Returns:
          Tensor of shape [B, N, U, W, C].
        """
        # term_bd_before_shift shape: [B, N, U, W, F_span]
        # Target shape after shift:  [B, N, U, W, C]

        # Padding amount for the last dimension (F_span) to become (C + 1)
        # C = key_context_size
        # F_span = max_span_plus_1
        pad_amount_last_dim = (key_context_size + 1) - max_span_plus_1

        # PyTorch F.pad expects (pad_left, pad_right, pad_top, pad_bottom ...)
        # We only pad the last dimension on the right.
        padding_tuple = (0, pad_amount_last_dim)

        term_bd_padded = nn.functional.pad(term_bd_before_shift, padding_tuple)
        # Shape after pad: [B, N, U, W, C+1]

        # Reshape for slicing (emulating JAX's behavior)
        # [B, N, U, W * (C+1)]
        term_bd_reshaped = term_bd_padded.reshape(
            (
                batch_size,
                num_heads,
                num_query_blocks,
                query_block_size * (key_context_size + 1),
            )
        )

        # Slice to effective [B, N, U, W * C]
        term_bd_sliced = term_bd_reshaped[:, :, :, : query_block_size * key_context_size]

        # Reshape back to [B, N, U, W, C]
        term_bd_shifted = term_bd_sliced.reshape(
            (
                batch_size,
                num_heads,
                num_query_blocks,
                query_block_size,
                key_context_size,
            )
        )
        return term_bd_shifted

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        # queries: [B, U, W, N, H] (batch, num_query_blocks, query_block_size, num_heads, head_dim)
        # keys:    [B, U, C, N, H] (batch, num_query_blocks, key_context_size, num_heads, head_dim)
        # C = W + L + R (key_context_size)
        # F_span = L + R + 1 (max_span + 1)

        batch_size, num_query_blocks, query_block_size, num_heads, head_dim = queries.shape
        _, _, key_context_size, _, _ = keys.shape

        # Relative positions for sinusoidal embeddings: [L, L-1, ..., -R]
        # Length is L+R+1 = self.max_span + 1
        pos_indices = torch.arange(self.max_backward, -self.max_forward - 1, -1, device=queries.device).unsqueeze(
            0
        )  # Shape [1, F_span]

        max_span_plus_1 = pos_indices.shape[1]  # F_span

        sin_emb_timing_signal = self._get_timing_signal_1d_pos(
            pos_indices, dtype=queries.dtype
        )  # Shape [1, F_span, self.channels]

        # Project sinusoidal embeddings: [1, F_span, self.channels] -> [1, F_span, N*H]
        projected_sin_emb = self.pos_proj(sin_emb_timing_signal)
        # Reshape to [1, F_span, N, H] then squeeze to [F_span, N, H]
        sin_emb = projected_sin_emb.reshape(1, max_span_plus_1, self.num_heads, self.head_dim).squeeze(
            0
        )  # Shape [F, N, H]

        # term_ac: Query-Key content interaction
        # queries: [B, U, W, N, H] -> permute to [B, N, U, W, H] for matmul
        # keys:    [B, U, C, N, H] -> permute to [B, N, U, H, C] for matmul
        queries_p = queries.permute(0, 3, 1, 2, 4)  # [B, N, U, W, H]
        keys_p_t = keys.permute(0, 3, 1, 4, 2)  # [B, N, U, H, C]
        term_ac = torch.matmul(queries_p, keys_p_t)  # [B, N, U, W, C]

        # term_bd: Query-Position interaction
        # Original einsum: term_bd_unshifed = torch.einsum('buwnh,fnh->bnuwf', queries, sin_emb)
        # queries shape: [B, U, W, N, H]
        # sin_emb shape: [F, N, H]
        # Target output shape: [B, N, U, W, F]

        # Permute queries to [B, N, U, W, H] for easier broadcasting with sin_emb
        q_permuted = queries.permute(0, 3, 1, 2, 4)

        # Permute sin_emb to [N, H, F] to prepare for matmul
        # sin_emb original is [F, N, H]
        s_permuted = sin_emb.permute(1, 2, 0)  # Shape: [N, H, F]

        # Reshape queries for matmul: [B, N, U*W, H]
        q_reshaped = q_permuted.reshape(batch_size, num_heads, num_query_blocks * query_block_size, head_dim)

        # Perform matmul: [B, N, U*W, H] @ [N, H, F]
        # s_permuted ([N, H, F]) will be broadcast to [B, N, H, F]
        # Result: [B, N, U*W, F]
        term_bd_unshifed_matmul = torch.matmul(q_reshaped, s_permuted)

        # Reshape to target [B, N, U, W, F]
        term_bd_unshifed = term_bd_unshifed_matmul.reshape(
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size,
            max_span_plus_1,
        )

        # Apply relative shift to term_bd_unshifed
        term_bd_shifted = self._relative_shift(
            term_bd_unshifed,
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size,
            key_context_size,
            max_span_plus_1,
        )  # Shape [B, N, U, W, C]

        return term_ac + term_bd_shifted


class Gemma3nAudioAttention(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig):
        super().__init__()
        self.config = config

        self.num_heads = self.config.conf_num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        self.chunk_size = self.config.conf_attention_chunk_size
        self.max_future_horizon = self.config.conf_attention_context_right
        self.max_past_horizon = max(0, self.config.conf_attention_context_left - 1)
        self.attention_logits_soft_cap = self.config.conf_attention_logit_cap
        self.context_size = self.chunk_size + self.max_past_horizon + self.max_future_horizon

        self.relative_position_embedding = Gemma3nAudioRelativePositionEmbedding(config)
        self.per_dim_scale = nn.Parameter(torch.zeros((self.head_dim,)))

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

        q_scale = self.head_dim**-0.5
        r_softplus_0 = 1.0 / torch.nn.functional.softplus(torch.tensor(0.0))
        self.register_buffer("q_scale", (q_scale * r_softplus_0).clone().detach(), persistent=False)

        lower_causal_mask = torch.tril(
            torch.ones((self.context_size, self.chunk_size), dtype=torch.bool),
            diagonal=0,
        ).T
        upper_causal_mask = torch.tril(
            torch.ones((self.chunk_size, self.context_size), dtype=torch.bool),
            diagonal=self.max_past_horizon + self.max_future_horizon,
        )
        local_causal_valid_mask = torch.ones((self.chunk_size, self.context_size), dtype=torch.bool)
        local_causal_valid_mask = local_causal_valid_mask * lower_causal_mask * upper_causal_mask
        self.register_buffer("local_causal_valid_mask", local_causal_valid_mask, persistent=False)

        self.register_buffer(
            "softcap",
            torch.tensor(self.attention_logits_soft_cap).float(),
            persistent=False,
        )

    def _pad_dim1(self, x: torch.Tensor, pad_left: int, pad_right: int) -> torch.Tensor:
        batch, _, *tail_shape = x.shape
        left = x.new_zeros((batch, pad_left, *tail_shape))
        right = x.new_zeros((batch, pad_right, *tail_shape))
        x = torch.cat([left, x, right], dim=1)
        return x

    def _convert_to_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Turns a sequence to non overlapping blocks.

        Args:
            hidden_states: a tensor of [batch, time, ...].

        Returns:
            A tensor of [batch, num_blocks, block_size, ...], with necessary
            paddings,
            where output[:, i, ...] are x[:, i*block_size:(i+1)*block_size, ...].
        """
        shape = hidden_states.shape
        b, t = shape[:2]
        num_blocks = (t + self.chunk_size - 1) // self.chunk_size

        if (padding_len := num_blocks * self.chunk_size - t) > 0:
            hidden_states = self._pad_dim1(hidden_states, 0, padding_len)

        permute_dims = (b, num_blocks, self.chunk_size) + shape[2:]
        hidden_states = hidden_states.reshape(permute_dims).contiguous()
        return hidden_states

    def _extract_block_context(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extracts temporal context for every block.

        Args:
            hidden_states: a tensor of [batch, time, ...].

        Returns:
            A tensor of [batch, num_blocks, context_size, ...], with necessary
            paddings,
            where context_size = block_size + left_context + right_context,
            and output[:, i, ...] are x[:, start-left_context:end+right_context,
            ...],
            start = i * block_size, end = (i + 1) * block_size.
        """
        pad_left = self.max_past_horizon
        # The JAX equivalent padding for signal.frame with pad_mode='valid' is
        # (left_context, right_context + block_size - 1) on the time dimension.
        # PyTorch's _pad_dim1 applies padding symmetrically if only one value is given,
        # or (pad_dim_start, pad_dim_end) if two are given.
        # Our _pad_dim1(x, pad_left, pad_right) pads dim -2 (time for [B,T,N,H])
        # or dim 1 (time for [B,T]).
        # The current pad_right calculation matches the JAX effective padding.
        pad_right = self.max_future_horizon + self.chunk_size - 1
        hidden_states = self._pad_dim1(hidden_states, pad_left, pad_right)

        frame_len = self.context_size
        frame_step = self.chunk_size

        # Directly use unfold without the subframe_factor logic
        # x.unfold(dimension, size, step)
        # dimension=1 (time dimension, assuming x is [B, T_padded, ...])
        # size=frame_len (context_size)
        # step=frame_step (chunk_size)
        x_unfolded = hidden_states.unfold(dimension=1, size=frame_len, step=frame_step)

        # If x was [B, T_padded], x_unfolded is [B, num_blocks, frame_len]
        # If x was [B, T_padded, N, H], x_unfolded is [B, num_blocks, N, H, frame_len]
        # We want to match JAX's typical output for such operations which might be
        # [B, num_blocks, frame_len, N, H] if N, H are present.
        # The relative_position_embedding expects keys as [B, U, C, N, H].
        # If x_unfolded is [B, U, N, H, C(frame_len)], we need to move C.
        if hidden_states.ndim > 2 and x_unfolded.ndim > 3:  # Check if inner dimensions (like N, H) exist
            # Current shape after unfold for [B, T_pad, N, H] is [B, U, N, H, C]
            # Target shape for keys in RPE: [B, U, C, N, H]
            x_unfolded = torch.movedim(x_unfolded, source=-1, destination=2)

        return x_unfolded.contiguous()

    def forward(self, hidden_states: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        # sl.Dense uses jax.numpy.einsum("...a,abcd->...bcd") and jax.numpy.select()
        qkv_shape = (*hidden_states.shape[:-1], self.num_heads, self.head_dim)
        query_states = self.q_proj(hidden_states).reshape(qkv_shape).contiguous()
        key_states = self.k_proj(hidden_states).reshape(qkv_shape).contiguous()
        value_states = self.v_proj(hidden_states).reshape(qkv_shape).contiguous()

        per_dim_scale_sp = torch.nn.functional.softplus(self.per_dim_scale)

        broadcast_shape = (1, 1, 1, self.head_dim)
        per_dim_scale_sp_broadcast = per_dim_scale_sp.view(broadcast_shape)
        query_states = query_states * self.q_scale * per_dim_scale_sp_broadcast

        batch_size, q_time = query_states.shape[:2]

        query_blocks = self._convert_to_block(query_states)
        key_blocks = self._extract_block_context(key_states)
        value_blocks = self._extract_block_context(value_states)
        num_query_blocks = query_blocks.shape[1]

        # 1. Create a mask indicating originally valid positions.
        original_valid_mask = ~mask  # True for valid, False for padded

        # 2. Extract blocks from this validity mask.
        extracted_valid_mask_blocks = self._extract_block_context(original_valid_mask)

        # If subframe_factor was used in _extract_block_context for a [B, T] input mask,
        # the shape might be [B, U, C/SF, SF]. Reshape to [B, U, C].
        # batch_size and num_query_blocks are known from query_blocks.
        # self.context_size is C.
        if (
            extracted_valid_mask_blocks.ndim == 4
            and extracted_valid_mask_blocks.shape[2] * extracted_valid_mask_blocks.shape[3] == self.context_size
        ):
            extracted_valid_mask_blocks = extracted_valid_mask_blocks.reshape(
                batch_size, num_query_blocks, self.context_size
            )
        # After potential reshape, ensure it's [B, U, C] if it was from a [B,T] mask.
        # This assertion might be too strict if _extract_block_context handles higher-rank inputs differently,
        # but for the mask case, this should hold.
        if extracted_valid_mask_blocks.shape != (
            batch_size,
            num_query_blocks,
            self.context_size,
        ):
            raise ValueError(
                "Shape of extracted_valid_mask_blocks"
                f" {extracted_valid_mask_blocks.shape} is not ({batch_size},"
                f" {num_query_blocks}, {self.context_size}) after potential reshape."
            )

        # 3. Expand dimensions for broadcasting with logits and causal mask.
        # Target shape for broadcasting with logits [B,N,U,W,C]
        # extracted_valid_mask_blocks to [B, 1, U, 1, C]
        condition_from_input_validity = extracted_valid_mask_blocks.unsqueeze(1).unsqueeze(-2)

        # self.local_causal_valid_mask is [W, C], True where allowed by local window.
        # Expand to [1, 1, 1, W, C]
        condition_from_causality = self.local_causal_valid_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # 4. Combine the two conditions.
        # final_condition will be True where a key is *both* originally valid *and* causally accessible.
        # Broadcasts to [B, 1, U, W, C]
        final_condition_for_where = torch.logical_and(
            condition_from_input_validity,
            condition_from_causality.to(condition_from_input_validity.device),  # Ensure same device
        )

        # Embed queries and keys
        logits = self.relative_position_embedding(query_blocks, key_blocks)

        # Apply attention logit softcap
        # Ensure softcap is on the same device as logits
        softcap_val = self.softcap.to(logits.device)
        logits = logits / softcap_val
        logits = torch.tanh(logits)
        logits = logits * softcap_val

        # Apply the combined mask.
        # final_condition_for_where will broadcast with logits [B,N,U,W,C]
        logits = torch.where(final_condition_for_where, logits, torch.finfo(logits.dtype).min)
        probabilities = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32).to(dtype=value_blocks.dtype)

        # context_vectors is adapted from jax.numpy.einsum("BNuwc,BucNH->BuwNH", ...)
        b_dim, n_dim, u_dim, w_dim, c_dim = probabilities.shape
        h_dim = value_blocks.shape[-1]
        prob_bun = probabilities.permute(0, 2, 1, 3, 4).reshape(-1, w_dim, c_dim)
        v_bun = value_blocks.permute(0, 1, 3, 2, 4).reshape(-1, c_dim, h_dim)
        result_bmm = torch.bmm(prob_bun, v_bun)
        context_vectors = result_bmm.reshape(b_dim, u_dim, n_dim, w_dim, h_dim).permute(0, 1, 3, 2, 4)
        context_vectors = context_vectors.reshape(
            (
                batch_size,
                num_query_blocks * self.chunk_size,
                self.num_heads,
                self.head_dim,
            )
        )
        context_vectors = context_vectors[:, :q_time]

        return context_vectors


class Gemma3nAudioCumulativeGroupNorm(nn.Module):
    """Applies Group Normalization cumulatively over the time dimension.

    This layer normalizes the input by calculating the mean and variance
    cumulatively over the time dimension (dim 1). The statistics are computed
    over all feature dimensions (specified by `feature_dims` and `num_channels`)
    for elements marked as valid by the optional `mask`.

    If a `mask` is provided (True for valid, False for invalid/padded),
    invalid time steps do not contribute to the statistics calculation, and
    their corresponding output values are zeroed out.

    Scale and bias, if enabled, are applied per-channel (last dimension).
    This behavior is similar to JAX's `GroupNormalization` with `num_groups=1`
    and `cumulative=True`.
    """

    def __init__(
        self,
        num_channels: int,  # Number of channels (size of the last dimension)
        feature_dims: Sequence[int],  # Sizes of non-channel feature dimensions, e.g., (H, W) for input [B,T,H,W,C]
        eps: float = 1e-3,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.feature_dims = tuple(feature_dims)
        self.eps = eps

        # Scale parameter depends only on the channel dimension
        self.weight = nn.Parameter(torch.ones(num_channels))

        # Axes for normalization: all dimensions except Batch (0) and Time (1).
        # For input [B, T, *feature_dims, C], these are dims from 2 onwards.
        self.reduction_axes = tuple(range(2, 2 + len(self.feature_dims) + 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Applies cumulative group norm, optionally using a mask.

        Args:
          hidden_states: Input tensor, shape [B, T, *feature_dims, C].

        Returns:
          Normalized tensor with the same shape as x.
        """
        expected_input_suffix = self.feature_dims + (self.num_channels,)
        if hidden_states.shape[2:] != expected_input_suffix:
            raise ValueError(
                f"Input tensor shape suffix {hidden_states.shape[2:]} does not match expected"
                f" suffix (feature_dims + num_channels) {expected_input_suffix}"
            )

        input_dtype = hidden_states.dtype
        # Calculations are performed in float32 for numerical stability.
        calc_dtype = torch.float32
        x_calc = hidden_states.to(calc_dtype)

        # Prepare a broadcastable mask (`mask_calc`).
        # If no mask is provided, treat all elements as valid
        # (mask_calc is all ones).
        # Otherwise, expand the [B, T] mask to [B, T, 1, ..., 1] for broadcasting.
        mask_calc = torch.ones_like(x_calc, dtype=calc_dtype)

        # Cumulative Statistics Calculation
        # 1. Sum of values over reduction axes at each time step.
        sum_values_at_t = torch.sum(x_calc, dim=self.reduction_axes, keepdim=True)
        # 2. Cumulative sum of values over time.
        cum_sum_values = torch.cumsum(sum_values_at_t, dim=1)

        # 3. Count of valid elements in the normalization group at each time step.
        #    (A "group" here consists of all features at a given Batch, Time).
        elements_in_group_at_t = torch.sum(mask_calc, dim=self.reduction_axes, keepdim=True)
        # 4. Cumulative count of valid elements over time.
        cum_count_elements = torch.cumsum(elements_in_group_at_t, dim=1)
        # Avoid division by zero if all preceding elements were masked.
        safe_cum_count_elements = torch.clamp(cum_count_elements, min=1.0)

        # 5. Cumulative mean.
        cum_mean = cum_sum_values / safe_cum_count_elements

        # 6. Sum of squared differences from the cumulative mean.
        #    Only sum for valid elements: (x_calc - cum_mean)^2 * mask_calc.
        #    Using x_calc here for the difference, as cum_mean already accounts for masking.
        squared_diff_from_mean = (x_calc - cum_mean).pow(2)
        sum_sq_diff_at_t = torch.sum(squared_diff_from_mean, dim=self.reduction_axes, keepdim=True)

        # 7. Cumulative sum of squared differences over time.
        cum_sum_sq_diff = torch.cumsum(sum_sq_diff_at_t, dim=1)

        # 8. Cumulative variance.
        cum_variance = cum_sum_sq_diff / safe_cum_count_elements

        # Normalize the input using the calculated cumulative statistics:
        # (x - E[x]) / sqrt(Var[x] + eps)
        normalized_x = (x_calc - cum_mean) * torch.rsqrt(cum_variance + self.eps)

        # Apply affine transformation (scale and bias) if enabled.
        # Scale and bias are applied per-channel (last dimension).
        scale = self.weight.to(calc_dtype)
        # Reshape for broadcasting: [C] -> [1, ..., 1, C]
        scale_view_shape = [1] * (hidden_states.dim() - 1) + [self.num_channels]
        normalized_x = normalized_x * scale.view(scale_view_shape)

        # Zero out outputs for time steps that were originally masked (where mask_calc is 0).
        # This ensures padded/invalid positions in the input result in zero output.
        final_output = normalized_x * mask_calc

        return final_output.to(input_dtype)


class Gemma3nAudioSSCPConvBlock(nn.Module):
    """A single convolution block for the SubSampleConvProjection.

    This block consists of a 2D convolution, followed by CumulativeGroupNorm,
    and a ReLU activation. It handles manual padding for the convolution.
    """

    def __init__(
        self,
        config: Gemma3nAudioConfig,
        idx: int,
        input_freq_dim: int,  # Changed from input_spatial_dim
        manual_padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    ):
        super().__init__()
        self.config = config
        self.manual_padding = manual_padding

        # in_channels is 1 for the first block, or C_out from previous block's conv
        in_channels = 1 if idx == 0 else self.config.sscp_conv_channel_size[idx - 1]
        out_channels = self.config.sscp_conv_channel_size[idx]
        kernel_h, kernel_w = self.config.sscp_conv_kernel_size[idx]
        stride_h, stride_w = self.config.sscp_conv_stride_size[idx]

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(
                kernel_h,
                kernel_w,
            ),  # Kernel (kH, kW) operates on (Time, Freq_dim)
            stride=(stride_h, stride_w),
            padding=(0, 0),  # Manual padding is used
            bias=False,
        )

        # Calculate output frequency dimension (f_out_conv) after this convolution.
        # input_freq_dim is the unpadded width (feature dimension).
        # self.manual_padding is (pad_F_left, pad_F_right, pad_T_top, pad_T_bottom)
        f_in_padded = input_freq_dim + self.manual_padding[0] + self.manual_padding[1]
        f_out_conv = (f_in_padded - kernel_w) // stride_w + 1

        self.norm = Gemma3nAudioCumulativeGroupNorm(
            num_channels=out_channels,  # Channels of the conv output
            feature_dims=(f_out_conv,),  # The frequency dimension size after conv
            eps=self.config.sscp_conv_group_norm_eps,
        )

        self.activation = nn.ReLU()

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:
        # Input audio_encodings is [B, C_in, T_in, F_in] (e.g., C_in=1)
        # manual_padding is (pad_F_left, pad_F_right, pad_T_top, pad_T_bottom)
        # F.pad applies to last two dims: F_in then T_in
        audio_encodings_padded = F.pad(audio_encodings, self.manual_padding, mode="constant", value=0.0)
        # Expected padded shape for F_in, k_w=3, pad_F=(1,1) -> F_padded = F_in+2
        # Expected padded shape for T_in, k_h=3, pad_T=(0,2) -> T_padded = T_in+2
        audio_encodings_conv = self.conv(audio_encodings_padded)
        # Expected conv output shape: [B, C_out, T_out, F_out]
        # Input to norm is [B, T_out, F_out, C_out]
        x_for_norm = audio_encodings_conv.permute(0, 2, 3, 1).contiguous()
        x_normed = self.norm(x_for_norm)
        # Output of norm is [B, T_out, F_out, C_out], permute back to [B, C_out, T_out, F_out]
        audio_encodings_normed = x_normed.permute(0, 3, 1, 2).contiguous()
        return self.activation(audio_encodings_normed)


class Gemma3nAudioSubSampleConvProjection(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig):
        super().__init__()
        self.config = config

        current_f_for_block_input = config.input_feat_size  # Start with original feature dim
        calculated_block_padding = []
        calculated_f_out_dims = []  # Tracking frequency dimension output sizes

        for i in range(2):  # Assuming 2 conv layers as per sscp_conv_... arrays
            kernel_h, kernel_w = config.sscp_conv_kernel_size[i]
            stride_h, stride_w = config.sscp_conv_stride_size[i]

            # Padding for Time (Height for Conv2d) - REVERSE_CAUSAL like
            # JAX 'reverse_causal' padding is (0, kernel_size - 1)
            pad_t_top = 0
            pad_t_bottom = kernel_h - 1

            # Frequency Padding (Width for Conv2d)
            # Based on JAX effective padding (1,1) for F_in=10, K_w=3, S_w=2
            # and the successful test configuration.
            # If kernel/stride/input_freq for frequency changes, this might need re-evaluation
            # to match generic JAX 'SAME' behavior if it differs.
            pad_f_left = 1
            pad_f_right = 1

            manual_padding_tuple = (
                pad_f_left,
                pad_f_right,
                pad_t_top,
                pad_t_bottom,
            )
            calculated_block_padding.append(manual_padding_tuple)

            # Calculate output frequency dimension after this convolution
            # This uses the actual padding applied and kernel/stride.
            f_in_padded = current_f_for_block_input + pad_f_left + pad_f_right
            f_out_after_conv = (f_in_padded - kernel_w) // stride_w + 1  # Assuming dilation_w = 1
            calculated_f_out_dims.append(f_out_after_conv)
            current_f_for_block_input = f_out_after_conv

        self.conv_0 = Gemma3nAudioSSCPConvBlock(
            idx=0,
            input_freq_dim=config.input_feat_size,  # Pass original feature dim
            config=config,
            manual_padding=calculated_block_padding[0],
        )
        self.conv_1 = Gemma3nAudioSSCPConvBlock(
            idx=1,
            input_freq_dim=calculated_f_out_dims[0],  # Output freq dim from conv_0
            config=config,
            manual_padding=calculated_block_padding[1],
        )
        final_c_out = config.sscp_conv_channel_size[-1]
        final_f_out = calculated_f_out_dims[-1]  # Final frequency dimension
        self.input_proj_in_features = final_c_out * final_f_out
        self.input_proj_linear = nn.Linear(self.input_proj_in_features, self.config.hidden_size, bias=False)

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:
        # audio_encodings is [B, T, F_in]
        # Reshape to [B, 1, T, F_in] (Batch, Channels=1, Height=Time, Width=F_in)
        audio_encodings_reshaped = audio_encodings.unsqueeze(1)
        x = self.conv_0(audio_encodings_reshaped)
        x = self.conv_1(x)
        # x from conv_1 is [B, C_out_1, T_out_1, F_out_1]
        b, c_out, t_out, f_out = x.shape
        # Permute to [B, T_out_1, F_out_1, C_out_1] then flatten F_out_1 and C_out_1
        x_permuted = x.permute(0, 2, 3, 1).contiguous()
        output_flattened = x_permuted.view(b, t_out, f_out * c_out)
        output = self.input_proj_linear(output_flattened)
        return output


class Gemma3nAudioConformerAttention(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig):
        super().__init__()
        self.config = config
        self.post_in_features = self.config.hidden_size
        self.register_buffer("gradient_clipping", torch.tensor(self.config.gradient_clipping), persistent=False)
        self.pre_attn_norm = Gemma3nRMSNorm(self.config.hidden_size)
        self.attn = Gemma3nAudioAttention(config)
        self.post = nn.Linear(self.post_in_features, self.config.hidden_size, bias=False)
        self.post_norm = Gemma3nRMSNorm(self.config.hidden_size)

    def forward(self, audio_encodings: torch.Tensor, audio_mel_mask: torch.BoolTensor) -> torch.Tensor:
        audio_encodings_input_to_attn = audio_encodings
        audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping, self.gradient_clipping)
        audio_encodings_norm = self.pre_attn_norm(audio_encodings)
        # Output of self.attn is [B, T, NumHeads, HeadDim]
        audio_encodings_attn_out = self.attn(audio_encodings_norm, audio_mel_mask)

        # Reshape from [B, T, NumHeads, HeadDim] to [B, T, NumHeads * HeadDim]
        # NumHeads * HeadDim = hidden_size
        b, t, num_heads, head_dim = audio_encodings_attn_out.shape
        audio_encodings_reshaped = audio_encodings_attn_out.reshape(b, t, num_heads * head_dim)

        audio_encodings = self.post(audio_encodings_reshaped)
        audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping, self.gradient_clipping)
        return audio_encodings_input_to_attn + self.post_norm(audio_encodings)


class Gemma3nAudioConformerFeedForward(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig):
        super().__init__()
        self.config = config

        self.register_buffer("gradient_clipping", torch.tensor(self.config.gradient_clipping), persistent=False)

        self.pre_layer_norm = Gemma3nRMSNorm(self.config.hidden_size)
        self.ffw_layer_1 = nn.Linear(self.config.hidden_size, self.config.hidden_size * 4, bias=False)
        self.ffw_layer_2 = nn.Linear(self.config.hidden_size * 4, self.config.hidden_size, bias=False)
        self.post_layer_norm = Gemma3nRMSNorm(self.config.hidden_size)
        self.post_layer_scale = torch.tensor(self.config.conf_residual_weight)

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:
        residual = audio_encodings
        audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping, self.gradient_clipping)
        audio_encodings = self.pre_layer_norm(audio_encodings)
        audio_encodings: torch.Tensor = self.ffw_layer_1(audio_encodings)
        audio_encodings = nn.functional.silu(audio_encodings)
        audio_encodings: torch.Tensor = self.ffw_layer_2(audio_encodings)
        audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping, self.gradient_clipping)
        audio_encodings = self.post_layer_norm(audio_encodings)
        return residual + (audio_encodings * self.post_layer_scale)


class Gemma3nAudioConformerLightConv1d(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig):
        super().__init__()
        self.config = config

        self.pre_layer_norm = Gemma3nRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.linear_start = nn.Linear(self.config.hidden_size, self.config.hidden_size * 2, bias=False)
        self.depthwise_conv1d = nn.Conv1d(
            in_channels=self.config.hidden_size,
            out_channels=self.config.hidden_size,
            kernel_size=self.config.conf_conv_kernel_size,
            stride=1,
            padding=0,  # Manual causal padding
            groups=self.config.hidden_size,  # Depthwise
            bias=False,
        )
        self.register_buffer("gradient_clipping", torch.tensor(self.config.gradient_clipping), persistent=False)
        self.conv_norm = Gemma3nRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.linear_end = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)

        self.causal_padding = self.config.conf_conv_kernel_size - 1

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:
        audio_encodings_residual = audio_encodings  # Save for residual connection

        audio_encodings = self.pre_layer_norm(audio_encodings)
        audio_encodings = self.linear_start(audio_encodings)
        audio_encodings = torch.nn.functional.glu(audio_encodings, dim=-1)
        # Permute for Conv1d: [B, T, D] -> [B, D, T]
        audio_encodings_permuted = audio_encodings.permute(0, 2, 1)
        # Apply manual causal padding
        audio_encodings_permuted_padded = F.pad(audio_encodings_permuted, (self.causal_padding, 0))
        audio_encodings = self.depthwise_conv1d(audio_encodings_permuted_padded)
        # Permute back: [B, D, T_out] -> [B, T_out, D]
        audio_encodings = audio_encodings.permute(0, 2, 1)
        audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping, self.gradient_clipping)
        audio_encodings = self.conv_norm(audio_encodings)
        audio_encodings = nn.functional.silu(audio_encodings)
        audio_encodings = self.linear_end(audio_encodings)
        output = audio_encodings + audio_encodings_residual
        return output


class Gemma3nAudioConformerBlock(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig):
        super().__init__()
        self.config = config

        self.ffw_layer_start = Gemma3nAudioConformerFeedForward(self.config)
        self.attention = Gemma3nAudioConformerAttention(self.config)
        self.lconv1d = Gemma3nAudioConformerLightConv1d(self.config)
        self.ffw_layer_end = Gemma3nAudioConformerFeedForward(self.config)
        self.register_buffer("gradient_clipping", torch.tensor(self.config.gradient_clipping), persistent=False)
        self.norm = Gemma3nRMSNorm(self.config.hidden_size)

    def forward(self, audio_encodings: torch.Tensor, audio_mel_mask: torch.BoolTensor) -> torch.Tensor:
        audio_encodings = self.ffw_layer_start(audio_encodings)
        audio_encodings = self.attention(audio_encodings, audio_mel_mask)
        validity_mask_for_lconv = ~audio_mel_mask  # True for valid
        audio_encodings_for_lconv_input = audio_encodings * validity_mask_for_lconv.unsqueeze(-1).to(
            audio_encodings.dtype
        )
        audio_encodings = self.lconv1d(audio_encodings_for_lconv_input)

        audio_encodings = self.ffw_layer_end(audio_encodings)
        audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping, self.gradient_clipping)
        output = self.norm(audio_encodings)
        return output


class Gemma3nAudioEncoder(PreTrainedModel):
    """An audio encoder based on the [Universal Speech Model](https://arxiv.org/abs/2303.01037) architecture."""

    config: Gemma3nAudioConfig

    main_input_name = "audio_mel"

    def __init__(self, config: Gemma3nAudioConfig):
        super().__init__(config)
        self.config = config

        self.subsample_conv_projection = Gemma3nAudioSubSampleConvProjection(config)
        self.conformer = nn.ModuleList(
            [Gemma3nAudioConformerBlock(config) for _ in range(config.conf_num_hidden_layers)]
        )

    def forward(
        self, audio_mel: torch.Tensor, audio_mel_mask: torch.BoolTensor
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        """Encodes a batch of MELs.

        Args:
            audio_mel: a torch.Tensor of shape [batch, num_frames, num_channels,
              mel_bins].

        Returns:
            audio_encodings: a torch.Tensor of shape
                `[batch_size, self.config.audio_soft_tokens_per_image,
                self.config.audio_config.hidden_size]`
            audio_mel_mask: a torch.BoolTensor of shape [batch, num_frames].
        """
        audio_encodings = self.subsample_conv_projection(audio_mel)  # audio_encodings: [B, T_sub, D]

        # Subsample the input audio_mel_mask to match the time dimension of audio_encodings (T_sub)
        t_sub = audio_encodings.shape[1]

        time_stride_product = 1
        for stride_pair_idx in range(len(self.config.sscp_conv_stride_size)):
            time_stride_product *= self.config.sscp_conv_stride_size[stride_pair_idx][0]

        # Create indices for gathering from the original mask.
        # These indices map to original time steps corresponding to the start of each
        # receptive field in the subsampled output.
        indices = torch.arange(t_sub, device=audio_mel_mask.device) * time_stride_product
        indices = torch.clamp(indices, max=audio_mel_mask.shape[1] - 1)  # Ensure indices are valid

        # Expand indices for batch compatibility if B > 1 and indices is 1D.
        if audio_mel_mask.ndim > 1 and indices.ndim == 1:
            indices = indices.unsqueeze(0).expand(audio_mel_mask.shape[0], -1)  # [B, T_sub]
        elif (
            audio_mel_mask.ndim == indices.ndim
            and audio_mel_mask.shape[0] == 1
            and indices.shape[0] != 1
            and t_sub == indices.shape[0]
        ):
            # Handle case where B=1 but indices became [T_sub] instead of [1, T_sub]
            indices = indices.unsqueeze(0)

        current_mask = torch.gather(audio_mel_mask, 1, indices)  # [B, T_sub]

        for block in self.conformer:
            audio_encodings = block(audio_encodings, current_mask)  # Pass the processed mask

        if self.config.conf_reduction_factor > 1:
            audio_encodings = audio_encodings[:, :: self.config.conf_reduction_factor]
            # Reduce the mask as well
            current_mask = current_mask[:, :: self.config.conf_reduction_factor]

        audio_encodings = audio_encodings.masked_fill(current_mask.unsqueeze(-1), 0.0)
        return audio_encodings, current_mask


# ==== Language Model ====


class Gemma3nTextScaledWordEmbedding(Gemma3TextScaledWordEmbedding):
    pass


class Gemma3nTextLaurelBlock(nn.Module):
    """Learned Augmented Residual Layer"""

    def __init__(self, config: Gemma3nTextConfig):
        super().__init__()
        self.config = config

        self.linear_left = nn.Linear(self.config.hidden_size, self.config.laurel_rank, bias=False)
        self.linear_right = nn.Linear(self.config.laurel_rank, self.config.hidden_size, bias=False)
        self.post_laurel_norm = Gemma3nRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        laurel_hidden_states: torch.Tensor = self.linear_left(hidden_states)
        laurel_hidden_states: torch.Tensor = self.linear_right(laurel_hidden_states)
        normed_laurel_hidden_states = self.post_laurel_norm(laurel_hidden_states)
        return hidden_states + normed_laurel_hidden_states


class Gemma3nTextMLP(Gemma2MLP):
    def __init__(self, config: Gemma3nTextConfig, layer_idx: int = 0):
        super().__init__(config)
        self.intermediate_size = config.intermediate_size[layer_idx]
        self.activation_sparsity = config.activation_sparsity_pattern[layer_idx]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_proj = self.gate_proj(hidden_states)
        if self.activation_sparsity > 0.0:
            gate_proj = self._gaussian_topk(gate_proj)
        activations = self.act_fn(gate_proj)
        up_proj = self.up_proj(hidden_states)
        down_proj = self.down_proj(activations * up_proj)
        return down_proj

    def _gaussian_topk(self, inputs: torch.Tensor) -> torch.Tensor:
        target_sparsity_tensor = torch.tensor(self.activation_sparsity, dtype=torch.float32, device=inputs.device)
        # normal_dist and std_multiplier are adapted from jax.scipy.stats.norm.ppf().
        #
        # References:
        #   *   https://docs.jax.dev/en/latest/_autosummary/jax.scipy.stats.norm.ppf.html
        #   *   https://pytorch.org/docs/stable/distributions.html#torch.distributions.normal.Normal
        #   *   https://pytorch.org/docs/stable/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.icdf
        normal_dist = torch.distributions.normal.Normal(0, 1)
        std_multiplier: torch.Tensor = normal_dist.icdf(target_sparsity_tensor)
        std_multiplier = std_multiplier.type(inputs.dtype)
        inputs_mean = torch.mean(inputs, dim=-1, keepdim=True)
        inputs_std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)
        cutoff_x = inputs_mean + inputs_std * std_multiplier
        return nn.functional.relu(inputs - cutoff_x)


class Gemma3nTextAltUp(nn.Module):
    """Alternating Updates (AltUp)

    The AltUp module wraps transformer layers. The `predict` step modifies the
    input to the transformer layer, and the `correct` step propagates the output
    of the transformer layer to the sparsely updated dimensions.

    See more in the research paper:

    https://proceedings.neurips.cc/paper_files/paper/2023/file/f2059277ac6ce66e7e5543001afa8bb5-Paper-Conference.pdf
    """

    def __init__(self, config: Gemma3nTextConfig):
        super().__init__()
        self.config = config
        self.correct_output_scale = nn.Parameter(torch.zeros(self.config.hidden_size))
        self.correction_coefs = nn.Linear(self.config.altup_num_inputs, self.config.altup_num_inputs, bias=False)
        self.prediction_coefs = nn.Linear(self.config.altup_num_inputs, self.config.altup_num_inputs**2, bias=False)
        self.modality_router = nn.Linear(self.config.hidden_size, self.config.altup_num_inputs, bias=False)
        self.router_norm = Gemma3nRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.register_buffer("router_input_scale", torch.tensor(self.config.hidden_size**-1.0), persistent=False)

    def compute_router_modalities(self, x: torch.Tensor) -> torch.Tensor:
        router_inputs = self.router_norm(x) * self.router_input_scale
        routed = self.modality_router(router_inputs)
        return torch.tanh(routed.float()).type_as(x)

    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predicts the output of a layer using a trainable map.

        Args:
            hidden_states: A 4D tensor of shape `[num_altup_inputs, batch_size, num_tokens, hidden_size]` derived by
                stacking the input embeddings and preprocessing the last `num_altup_inputs - 1` matrices.

        Returns:
            A 4D tensor of shape `[num_altup_inputs, batch_size, num_tokens, hidden_size]` containing the predictions.
        """
        modalities = self.compute_router_modalities(hidden_states[self.config.altup_active_idx])

        if self.training and self.config.altup_coef_clip is not None:
            self.prediction_coefs.weight.data.clamp_(-self.config.altup_coef_clip, self.config.altup_coef_clip)

        # Project and then transpose all 2D matrices contained so that mulmat gives the correct result
        all_coefs: torch.Tensor = (
            self.prediction_coefs(modalities)
            .reshape(*modalities.shape[:-1], self.config.altup_num_inputs, self.config.altup_num_inputs)
            .permute(0, 1, 3, 2)
        )

        # permute hidden_states to [batch_size, num_tokens, hidden_size, altup_num_inputs]
        predictions = torch.matmul(hidden_states.permute(1, 2, 3, 0), all_coefs)
        predictions = predictions.permute(3, 0, 1, 2)  # undo the permute
        predictions += hidden_states  # add the original input
        return predictions.contiguous().type_as(hidden_states)

    def correct(self, predictions: torch.Tensor, activated: torch.Tensor) -> torch.Tensor:
        """Corrects the predictions relative to the

        Args:
            predictions: A 4D tensor of shape `[num_altup_inputs, batch_size, num_tokens, hidden_size]` derived by
                stacking the input embeddings and preprocessing the last `num_altup_inputs - 1` matrices.
            activated: A 3D tensor of shape `[batch_size, num_tokens, hidden_size]` containing the activated inputs.

        Returns:
            A 4D tensor of shape `[num_altup_inputs, batch_size, num_tokens, hidden_size]` correcting the original
                predictions relative to the activated input embeddings.
        """
        modalities = self.compute_router_modalities(activated)
        innovation = activated - predictions[self.config.altup_active_idx]  # (batch, num_tokens, hidden_size)
        innovation = innovation.repeat(self.config.altup_num_inputs, 1, 1, 1)  # Repeat on dim0 to match predictions

        if self.config.altup_coef_clip is not None:
            self.correction_coefs.weight.data.clamp_(-self.config.altup_coef_clip, self.config.altup_coef_clip)

        # all_coefs adapted from jax.numpy.einsum("...p,pi->...i", ...)
        # Permute to (altup_num_inputs, batch_size, num_tokens) as the last dim is a scalar applied to each altup input
        # and expand on dim1 for broadcastability
        all_coefs: torch.Tensor = self.correction_coefs(modalities) + 1.0
        all_coefs = all_coefs.permute(2, 0, 1).unsqueeze(-1)

        corrected = torch.mul(innovation, all_coefs)
        corrected += predictions  # add the original input
        return corrected.contiguous().type_as(activated)

    def forward(self, corrected: torch.Tensor) -> torch.Tensor:
        """
        This is only defined as the `forward` so that accelerate hooks can move correctly `correct_output_scale`
        (which is a nn.Parameter, not a Module) between devices when offloading. It is otherwise only used in
        `scale_corrected_output`
        """
        return (corrected.type_as(self.correct_output_scale) * self.correct_output_scale).type_as(corrected)

    def scale_corrected_output(self, corrected: torch.Tensor) -> torch.Tensor:
        """Scales the provided 3D tensor of shape [batch_size, num_tokens, hidden_size]."""
        return self.forward(corrected)


class Gemma3nTextRotaryEmbedding(Gemma2RotaryEmbedding):
    pass


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        x (`torch.Tensor`): The tensor to embed.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


class Gemma3nTextAttention(Gemma3Attention):
    def __init__(self, config: Gemma3nTextConfig, layer_idx: int):
        super().__init__()
        del self.attn_logit_softcapping
        del self.scaling
        self.v_norm = Gemma3nRMSNorm(dim=config.head_dim, eps=config.rms_norm_eps, with_scale=False)

        first_kv_shared_layer_idx = self.config.num_hidden_layers - self.config.num_kv_shared_layers
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        # Find the index of the last sliding or full layer before sharing starts (or None if no sharing)
        layer_type = config.layer_types[layer_idx]
        self.kv_shared_layer_index = (
            first_kv_shared_layer_idx - 1 - config.layer_types[first_kv_shared_layer_idx - 1 :: -1].index(layer_type)
            if self.is_kv_shared_layer
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.config.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        if self.is_kv_shared_layer and self.kv_shared_layer_index is not None and past_key_value is not None:
            # Device of past layer may be different from current one
            indices = cache_position.to(past_key_value.layers[self.kv_shared_layer_index].keys.device)
            # In this case we need special handling of the slice as the layer is of fixed small size (for full layers, we never go beyond)
            if isinstance(past_key_value, HybridCache) and self.is_sliding:
                max_length = past_key_value.sliding_window
                indices = (
                    slice(0, max_length)
                    if cache_position.shape[0] > max_length
                    else cache_position.clamp(min=0, max=max_length - 1)
                )

            # Device of past layer may be different from current one
            key_states = past_key_value.layers[self.kv_shared_layer_index].keys[:, :, indices].to(query_states.device)
            value_states = (
                past_key_value.layers[self.kv_shared_layer_index].values[:, :, indices].to(query_states.device)
            )
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)

            value_states = self.v_proj(hidden_states).view(hidden_shape)
            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "sliding_window": self.sliding_window,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=1.0,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Gemma3nTextDecoderLayer(Gemma3DecoderLayer):
    def __init__(self, config: Gemma3nTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = Gemma3nTextMLP(config, layer_idx=layer_idx)

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.act_fn = ACT2FN[config.hidden_activation]

        self.altup = Gemma3nTextAltUp(config)
        self.laurel = Gemma3nTextLaurelBlock(config)
        self.self_attn = Gemma3nTextAttention(config, layer_idx)
        self.per_layer_input_gate = nn.Linear(self.hidden_size, self.hidden_size_per_layer_input, bias=False)
        self.per_layer_projection = nn.Linear(self.hidden_size_per_layer_input, self.hidden_size, bias=False)
        self.post_per_layer_input_norm = Gemma3nRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        per_layer_input: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        predictions = self.altup.predict(hidden_states)
        active_prediction = predictions[self.config.altup_active_idx]

        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_output = self.laurel(active_prediction_normed)

        # apply global RoPE to non-sliding layer only
        if self.self_attn.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

        attn, self_attn_weights = self.self_attn(
            hidden_states=active_prediction_normed,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        attn = self.post_attention_layernorm(attn)

        attn_gated = active_prediction + attn
        attn_laurel = (attn_gated + laurel_output) / math.sqrt(2)

        attn_norm = self.pre_feedforward_layernorm(attn_laurel)
        attn_ffw = self.mlp(attn_norm)
        attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
        attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm
        corrected_predictions = self.altup.correct(predictions, attn_ffw_laurel_gated)

        first_prediction = corrected_predictions[self.config.altup_active_idx].clone()
        if self.config.altup_correct_scale:
            first_prediction = self.altup.scale_corrected_output(first_prediction)

        # per_layer_input_gate adapted from jax.numpy.einsum("btd,dp->btp", ...)
        first_prediction = self.per_layer_input_gate(first_prediction)
        first_prediction = self.act_fn(first_prediction)
        first_prediction = torch.multiply(first_prediction, per_layer_input)

        # per_layer_projection adapted from jax.numpy.einsum("btp,pd->btd", ...)
        first_prediction = self.per_layer_projection(first_prediction)
        first_prediction = self.post_per_layer_input_norm(first_prediction)
        corrected_predictions[1:] += first_prediction

        outputs = (corrected_predictions,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Gemma3nPreTrainedModel(Gemma2PreTrainedModel):
    config: Gemma3nConfig
    base_model_prefix = ""
    _no_split_modules = ["Gemma3nTextDecoderLayer"]

    def _init_weights(self, module):
        Gemma2PreTrainedModel._init_weights(module)
        if isinstance(module, Gemma3nAudioCumulativeGroupNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, Gemma3nAudioAttention):
            module.per_dim_scale.data.zero_()
        elif isinstance(module, Gemma3nTextAltUp):
            module.correct_output_scale.data.zero_()


@auto_docstring(custom_intro="The base Gemma 3n language model without a language modeling head.")
class Gemma3nTextModel(Gemma3TextModel):
    config: Gemma3nTextConfig

    def __init__(self, config: Gemma3nTextConfig):
        super().__init__(config)

        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input

        self.embed_tokens_per_layer = Gemma3nTextScaledWordEmbedding(
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            self.padding_idx,
            embed_scale=config.hidden_size_per_layer_input**0.5,
        )

        self.per_layer_model_projection = nn.Linear(
            self.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )

        self.per_layer_projection_norm = Gemma3nRMSNorm(config.hidden_size_per_layer_input, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [Gemma3nTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = Gemma3nRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.altup_projections = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(1, self.config.altup_num_inputs)]
        )

        self.altup_unembed_projections = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(1, self.config.altup_num_inputs)]
        )

        self.register_buffer("per_layer_projection_scale", torch.tensor(self.hidden_size**-0.5), persistent=False)
        self.register_buffer("per_layer_input_scale", torch.rsqrt(torch.tensor(2.0)), persistent=False)
        self.rotary_emb = Gemma3nTextRotaryEmbedding(config=config)

        # TODO (raushan): Fix this after RoPE refactor. For now we hack it by
        # reassigning thetas when we want to create a local RoPE layer. Config
        # defaults should hold values for global RoPE.
        config = copy.deepcopy(config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3nTextRotaryEmbedding(config=config)

    def get_per_layer_inputs(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        per_layer_projection: torch.Tensor = self.per_layer_model_projection(inputs_embeds)
        per_layer_projection *= self.per_layer_projection_scale.to(
            dtype=inputs_embeds.dtype, device=per_layer_projection.device
        )
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        if per_layer_projection.shape != per_layer_inputs.shape:
            # per-layer inputs are sometimes padded with zeros, slice the relevant embeddings.
            per_layer_inputs = per_layer_inputs[..., : self.config.num_hidden_layers, :]

        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale.to(
            dtype=inputs_embeds.dtype, device=per_layer_projection.device
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        per_layer_inputs (torch.Tensor, *optional*, defaults to None):
            Pre-computed per-layer embeddings. If None, they are derived from input_ids if provided.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
            per_layer_inputs = self.get_per_layer_inputs(input_ids)

        per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        # embed positions
        hidden_states_0 = inputs_embeds

        # Initialize RoPE embeddings
        position_embeddings_global = self.rotary_emb(hidden_states_0, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states_0, position_ids)

        # Expand hidden_states to support per-layer inputs
        target_magnitude = torch.mean(hidden_states_0**2, dim=-1, keepdim=True) ** 0.5
        epsilon_tensor = torch.tensor(1e-5)

        temp_hidden_states = [hidden_states_0]
        for i in range(1, self.config.altup_num_inputs):
            # altup_proj adapted from jax.numpy.einsum("btp,pd->btd", ...)
            altup_proj = self.altup_projections[i - 1](hidden_states_0)
            current_hidden_state = altup_proj.to(dtype=hidden_states_0.dtype, device=target_magnitude.device)
            new_magnitude = torch.mean(current_hidden_state**2, dim=-1, keepdim=True)
            new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor.to(target_magnitude.device)))
            current_hidden_state = current_hidden_state * target_magnitude / new_magnitude
            temp_hidden_states.append(current_hidden_state)

        hidden_states = torch.stack(temp_hidden_states, dim=0)  # [num_altup_inputs, batch, seq_len, hidden_size]

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            causal_mask = causal_mask_mapping[decoder_layer.attention_type]
            per_layer_input = per_layer_inputs[:, :, decoder_layer.layer_idx, :]

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings_global,
                position_embeddings_local,
                per_layer_input,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # add hidden states from the last decoder layer (but before reprojecting to stay consistent with layer output)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Per-layer inputs to single output
        target_magnitude = torch.mean(hidden_states[0] ** 2, dim=-1, keepdim=True) ** 0.5
        temp_hidden_states = [hidden_states[0]]
        for i in range(1, self.config.altup_num_inputs):
            # altup_unembed_projections adapted from jax.numpy.einsum("btp,pd->btd", ...)
            altup_unemb_proj: torch.Tensor = self.altup_unembed_projections[i - 1](hidden_states[i])
            current_hidden_state = altup_unemb_proj.to(dtype=hidden_states_0.dtype, device=target_magnitude.device)
            new_magnitude = torch.mean(current_hidden_state**2, dim=-1, keepdim=True)
            new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor.to(target_magnitude.device)))
            current_hidden_state = current_hidden_state * target_magnitude / new_magnitude
            temp_hidden_states.append(current_hidden_state)

        hidden_states = torch.stack(temp_hidden_states)
        hidden_states = torch.mean(hidden_states, dim=0)
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@auto_docstring(custom_intro="The base Gemma 3n language model with a language modeling head.")
class Gemma3nForCausalLM(Gemma3ForCausalLM):
    _checkpoint_conversion_mapping = {"model.language_model": "model"}
    base_model_prefix = "model"


class Gemma3nMultimodalEmbedder(nn.Module):
    """Embeds token ids or soft tokens for multimodal content into language model space."""

    def __init__(
        self,
        multimodal_config: Union[Gemma3nAudioConfig, Gemma3nVisionConfig],
        text_config: Gemma3nTextConfig,
    ):
        super().__init__()

        self.multimodal_hidden_size = multimodal_config.hidden_size
        self.eps = multimodal_config.rms_norm_eps
        self.vocab_offset = multimodal_config.vocab_offset
        self.vocab_size = multimodal_config.vocab_size
        self.text_hidden_size = text_config.hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.multimodal_hidden_size)
        self.hard_embedding_norm = Gemma3nRMSNorm(self.multimodal_hidden_size, eps=self.eps)
        self.soft_embedding_norm = Gemma3nRMSNorm(self.multimodal_hidden_size, eps=self.eps)
        self.embedding_projection = nn.Linear(self.multimodal_hidden_size, self.text_hidden_size, bias=False)
        self.embedding_post_projection_norm = Gemma3nRMSNorm(self.text_hidden_size, eps=self.eps, with_scale=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embeds token ids or soft tokens for multimodal content into language model space.

        Args:
            input_ids: A torch.LongTensor containing the token ids to embed. Values should be in the range
                `[vocab_offset, vocab_offset + vocab_size)`.
            inputs_embeds: A torch.Tensor containing the soft tokens to embed.

        Returns:
            A torch.Tensor of embeddings with  shape `[batch_size, seq_len, self.config.text_config.hidden_size]`.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is not None:
            emb_norm = self.soft_embedding_norm(inputs_embeds)
        else:
            hard_emb = self.embedding(input_ids - self.vocab_offset)
            emb_norm = self.hard_embedding_norm(hard_emb)

        emb_norm_proj = self.embedding_projection(emb_norm)
        return self.embedding_post_projection_norm(emb_norm_proj)


@auto_docstring(
    custom_intro="""
    The base Gemma 3n model comprising a vision backbone, an audio backbone, and a language model without a
    language modeling head.
    """
)
class Gemma3nModel(PaliGemmaModel):
    _checkpoint_conversion_mapping = {}

    def __init__(self, config: Gemma3nConfig):
        super().__init__()
        del self.multi_modal_projector  # Replaced by Gemma3nVisionEmbedder
        self.vocab_size_per_layer_input = config.text_config.vocab_size_per_layer_input
        self.audio_tower = AutoModel.from_config(config.audio_config)
        self.embed_vision = Gemma3nMultimodalEmbedder(config.vision_config, config.text_config)
        self.embed_audio = Gemma3nMultimodalEmbedder(config.audio_config, config.text_config)

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Projects the last hidden state from the vision model into language model space.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.

        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values, do_pooling=False, return_dict=True
        ).last_hidden_state
        # Convert from (batch, channels, height, width) to (batch, height * width, channels) where:
        # height == width and height * width == Gemma3nConfig.vision_soft_tokens_per_image.
        vision_outputs = vision_outputs.reshape(
            vision_outputs.shape[0],
            self.config.vision_config.hidden_size,
            self.config.vision_soft_tokens_per_image,
        ).permute(0, 2, 1)
        # Normalize and embed the soft tokens into language model space.
        vision_outputs *= self.config.vision_config.hidden_size**0.5
        return self.embed_vision(inputs_embeds=vision_outputs)

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # text inputs
        pixel_values: Optional[torch.FloatTensor] = None,  # vision inputs
        input_features: Optional[torch.FloatTensor] = None,  # audio inputs
        attention_mask: Optional[torch.Tensor] = None,
        input_features_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **lm_kwargs,
    ) -> Gemma3nCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Gemma3nForConditionalGeneration

        >>> model = Gemma3nForConditionalGeneration.from_pretrained("google/gemma3n2-3b-mix-224")
        >>> processor = AutoProcessor.from_pretrained("google/gemma3n2-3b-mix-224")

        >>> prompt = "Where is the cat standing?"
        >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt,  return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs,)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Where is the cat standing?\nsnow"
        ```
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # Prepare per-layer inputs from inputs_ids
            per_layer_inputs_mask = torch.logical_and(input_ids >= 0, input_ids < self.vocab_size_per_layer_input)
            per_layer_inputs_tokens = torch.where(per_layer_inputs_mask, input_ids, torch.zeros_like(input_ids))
            per_layer_inputs = self.language_model.get_per_layer_inputs(per_layer_inputs_tokens)

            # Handle vision tokens (>= embed_vision.vocab_offset and < embed_audio.vocab_offset)
            vision_mask = torch.logical_and(
                input_ids >= self.embed_vision.vocab_offset, input_ids < self.embed_audio.vocab_offset
            )
            dummy_vision_token_id = self.embed_vision.vocab_offset + self.embed_vision.vocab_size - 1
            vision_input_ids = torch.where(vision_mask, input_ids, dummy_vision_token_id).to(inputs_embeds.device)
            vision_embeds = self.embed_vision(input_ids=vision_input_ids)
            expanded_vision_mask = vision_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = torch.where(expanded_vision_mask, vision_embeds, inputs_embeds)

            # Handle audio tokens (>= embed_audio.vocab_offset)
            audio_mask = input_ids >= self.embed_audio.vocab_offset
            dummy_audio_token_id = self.embed_audio.vocab_offset + self.embed_audio.vocab_size - 1
            audio_input_ids = torch.where(audio_mask, input_ids, dummy_audio_token_id).to(inputs_embeds.device)
            audio_embeds = self.embed_audio(input_ids=audio_input_ids)
            expanded_audio_mask = audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = torch.where(expanded_audio_mask, audio_embeds, inputs_embeds)
        else:
            per_layer_inputs = None

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            else:
                special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
                image_tokens_in_text = (special_image_mask).sum(dim=1).sum(dim=0)[0]
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text and "
                    f"{image_features.shape[0] * image_features.shape[1]} tokens from image embeddings."
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # Merge text and audio
        if input_features is not None and input_features_mask is not None:
            audio_features, audio_mask = self.get_audio_features(input_features, ~input_features_mask)

            # The Gemma3nProcessor expects all audio will be 30s in length and inserts 188 audio soft tokens into the
            # text to account for this. However, the audio preprocessing and encoder do not gurarantee they will
            # produce 188 soft tokens; they will produce at most that many tokens, but they may produce fewer tokens
            # depending on the length of the longest audio input in the batch. When we encounter this situation, we pad
            # the audio feature out to 188 soft tokens with the emebedding of the last token in the embed_audio vocab.
            audio_padding_toks = torch.tensor([[self.vocab_size - 1]], dtype=torch.long, device=audio_features.device)
            audio_padding_embs = self.embed_audio(input_ids=audio_padding_toks)
            audio_features = torch.where(audio_mask.unsqueeze(-1), audio_padding_embs, audio_features)

            audio_batch_size, audio_seq_len, audio_embed_dim = audio_features.shape
            extra_padding_tokens = self.config.audio_soft_tokens_per_image - audio_seq_len
            extra_padding_features = audio_padding_embs.expand(audio_batch_size, extra_padding_tokens, audio_embed_dim)

            audio_features = torch.cat((audio_features, extra_padding_features), dim=1)

            if input_ids is None:
                special_audio_mask = inputs_embeds == self.embed_audio(
                    input_ids=torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            else:
                special_audio_mask = (input_ids == self.config.audio_token_id).unsqueeze(-1)
                special_audio_mask = special_audio_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

            if not is_torchdynamo_compiling() and inputs_embeds[special_audio_mask].numel() != audio_features.numel():
                audio_tokens_in_text = (special_audio_mask).sum(dim=1).sum(dim=0)[0]
                raise ValueError(
                    f"Number of audio input features does not match number of special audio tokens in the input text. "
                    f"Got {audio_tokens_in_text} audio tokens in the text and "
                    f"{audio_features.shape[0] * audio_features.shape[1]} tokens from audio embeddings."
                )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_features)

        outputs = self.language_model(
            input_ids=None,
            per_layer_inputs=per_layer_inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **lm_kwargs,
        )

        return Gemma3nModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values if use_cache else None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
            audio_hidden_states=audio_features if input_features is not None else None,
        )

    def get_audio_features(
        self, input_features: torch.Tensor, input_features_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Projects the last hidden state from the audio encoder into language model space.

        Args:
            input_features (`torch.FloatTensor]` of shape `(num_images, seq_length, num_features)`):
               The tensors corresponding to the input audio.
            input_features_mask (`torch.FloatTensor]` of shape `(num_images, seq_length)`):
               The attention mask for the input audio.

        Returns:
            audio_features (`torch.Tensor`): Audio feature tensor of shape `(num_images, audio_length, embed_dim)`).
        """
        audio_outputs, audio_mask = self.audio_tower(input_features, input_features_mask)
        return self.embed_audio(inputs_embeds=audio_outputs), audio_mask

    def _update_causal_mask(self, **super_kwargs):
        raise AttributeError("We don't want to inherit it")


@auto_docstring(
    custom_intro="""
    The base Gemma 3n model comprising a vision backbone, an audio backbone, a language model, and a language modeling
    head.
    """
)
class Gemma3nForConditionalGeneration(PaliGemmaForConditionalGeneration):
    _checkpoint_conversion_mapping = {}
    base_model_prefix = "model"

    @property
    def audio_tower(self):
        return self.model.audio_tower

    @property
    def multi_modal_projector(self):
        raise AttributeError("Use embed_vision instead of multi_modal_projector.")

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # text inputs
        pixel_values: Optional[torch.FloatTensor] = None,  # vision inputs
        input_features: Optional[torch.FloatTensor] = None,  # audio inputs
        attention_mask: Optional[torch.Tensor] = None,
        input_features_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **lm_kwargs,
    ) -> Gemma3nCausalLMOutputWithPast:
        r"""
        input_features_mask (torch.Tensor, *optional*, defaults to None):
            The attention mask for the input audio.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels in
            `[0, ..., config.text_config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        >>> model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-4b-it")
        >>> processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

        >>> messages = [
        ...     {
        ...         "role": "system",
        ...         "content": [
        ...             {"type": "text", "text": "You are a helpful assistant."}
        ...         ]
        ...     },
        ...     {
        ...         "role": "user", "content": [
        ...             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
        ...             {"type": "text", "text": "Where is the cat standing?"},
        ...         ]
        ...     },
        ... ]

        >>> inputs = processor.apply_chat_template(
        ...     messages,
        ...     tokenizer=True,
        ...     return_dict=True,
        ...     return_tensors="pt",
        ...     add_generation_prompt=True
        ... )
        >>> # Generate
        >>> generate_ids = model.generate(**inputs)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "user\nYou are a helpful assistant.\n\n\n\n\n\nWhere is the cat standing?\nmodel\nBased on the image, the cat is standing in a snowy area, likely outdoors. It appears to"
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            input_features=input_features,
            attention_mask=attention_mask,
            input_features_mask=input_features_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            token_type_ids=token_type_ids,
            cache_position=cache_position,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **lm_kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if (final_logit_softcapping := self.config.get_text_config().final_logit_softcapping) is not None:
            logits = logits / final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * final_logit_softcapping

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()

            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)

        return Gemma3nCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            audio_hidden_states=outputs.audio_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        input_features=None,
        attention_mask=None,
        input_features_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        **kwargs,
    ):
        # Overwritten -- custom `position_ids` and `pixel_values` handling
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        # If we're in cached decoding stage, multimodal inputs should be None because input ids do not contain special
        # tokens anymore. Otherwise multimodal inputs should be passed to model.
        # NOTE: use_cache=False always needs pixel_values, input_features, and input_features_mask
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["input_features"] = input_features
            model_inputs["input_features_mask"] = input_features_mask

        return model_inputs

    def _prepare_4d_causal_attention_mask_with_cache_position(self, **super_kwargs):
        raise AttributeError("Do not inherit _prepare_4d_causal_attention_mask_with_cache_position from PaliGemma")


__all__ = [
    "Gemma3nAudioConfig",
    "Gemma3nAudioEncoder",
    "Gemma3nConfig",
    "Gemma3nForCausalLM",
    "Gemma3nForConditionalGeneration",
    "Gemma3nModel",
    "Gemma3nPreTrainedModel",  # noqa: F822
    "Gemma3nTextConfig",
    "Gemma3nTextModel",
    "Gemma3nVisionConfig",
]
