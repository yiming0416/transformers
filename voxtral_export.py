from transformers import VoxtralForConditionalGeneration, AutoProcessor
from transformers.cache_utils import DynamicCache
import torch
from torch.export import Dim, ShapesCollection
from torch.export._trace import _export

device = "cuda" if torch.cuda.is_available() else "cpu"
repo_id = "mistralai/Voxtral-Mini-3B-2507"

processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map=device)
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "url": "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/dude_where_is_my_car.wav",
            },
            {"type": "text", "text": "What can you tell me about this audio?"},
        ],
    }
]

inputs = processor.apply_chat_template(conversation)
inputs = inputs.to(device, dtype=torch.bfloat16)
#### export the full model ####
input_ids = inputs["input_ids"]
input_features = inputs["input_features"]
attention_mask = inputs["attention_mask"]
seq_length = input_ids.shape[1]
position_ids = torch.arange(0, seq_length, dtype=torch.int64, device=device).unsqueeze(0)
cache_position = torch.arange(0, seq_length, dtype=torch.int64, device=device)
sample_kwargs = {
    "input_ids": input_ids, 
    "input_features": input_features,
    "attention_mask": attention_mask, 
    "position_ids": position_ids,
    "past_key_values": DynamicCache(),
    "inputs_embeds": None,
    "labels": None,
    "use_cache": True, 
    "cache_position": cache_position,
    "logits_to_keep": 1
}

eager_out = model(**sample_kwargs)
past_key_values = eager_out.past_key_values

# specify dynamic shapes
shapes = ShapesCollection()
seq_len = Dim("seq_len")
for ix in range(len(past_key_values)):
    shapes[past_key_values.layers[ix].keys] = (Dim.STATIC, Dim.STATIC, seq_len, Dim.STATIC)
    shapes[past_key_values.layers[ix].values] = (Dim.STATIC, Dim.STATIC, seq_len, Dim.STATIC)

shapes[input_ids] = (Dim.STATIC, Dim.AUTO)
shapes[attention_mask] = (Dim.STATIC, Dim.AUTO)
shapes[position_ids] = (Dim.STATIC, Dim.AUTO)
shapes[cache_position] = (Dim.AUTO,)

# sample_kwargs["past_key_values"] = past_key_values

ep = _export(
    mod=model, 
    args=(), 
    kwargs=sample_kwargs, 
    pre_dispatch=False,
    dynamic_shapes=shapes,
    strict=False
)
ep_out = ep.module()(**sample_kwargs)

if torch.allclose(eager_out.logits, ep_out.logits):
    print("Exported model is correct!")
else:
    print("Exported model is incorrect!")

print("verify dynamic shape works for full model export")
conversation_1 = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "url": "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/dude_where_is_my_car.wav",
            },
            {"type": "text", "text": "This is a different prompt, what can you tell me about this audio from the link above?"},
        ],
    }
]

inputs = processor.apply_chat_template(conversation_1)
inputs = inputs.to(device, dtype=torch.bfloat16)
input_ids = inputs["input_ids"]
input_features = inputs["input_features"]
attention_mask = inputs["attention_mask"]
seq_length = input_ids.shape[1]
position_ids = torch.arange(0, seq_length, dtype=torch.int64, device=device).unsqueeze(0)
cache_position = torch.arange(0, seq_length, dtype=torch.int64, device=device)
sample_kwargs_1 = {
    "input_ids": input_ids, 
    "input_features": input_features,
    "attention_mask": attention_mask, 
    "position_ids": position_ids,
    "past_key_values": past_key_values,
    "inputs_embeds": None,
    "labels": None,
    "use_cache": True, 
    "cache_position": cache_position,
    "logits_to_keep": 1
}

eager_out_1 = model(**sample_kwargs_1)
ep_out_1 = ep.module()(**sample_kwargs_1)

if torch.allclose(eager_out.logits, ep_out.logits):
    print("Exported model is correct!")
else:
    print("Exported model is incorrect!")

#### export the language model ####
language_model = model.language_model

# manually create example inputs
attention_mask = torch.randint(low=0, high=2, size=(1, 1138), dtype=torch.int64, device=device)
position_ids = torch.arange(0, 1138, dtype=torch.int64, device=device).unsqueeze(0)
input_embeds = torch.rand((1, 1138, 3072), dtype=torch.bfloat16, device=device)
cache_position = torch.arange(0, 1138, dtype=torch.int64, device=device)
sample_kwargs = {
    "input_ids": None, 
    "attention_mask": attention_mask, 
    "position_ids": position_ids,
    "past_key_values": DynamicCache(),
    "inputs_embeds": input_embeds,
    "labels": None,
    "use_cache": True, 
    "cache_position": cache_position,
    "logits_to_keep": 1
}

eager_out = language_model(**sample_kwargs)
past_key_values = eager_out.past_key_values

# specify dynamic shapes
shapes = ShapesCollection()
seq_len = Dim("seq_len")
for ix in range(len(past_key_values)):
    shapes[past_key_values.layers[ix].keys] = (Dim.STATIC, Dim.STATIC, seq_len, Dim.STATIC)
    shapes[past_key_values.layers[ix].values] = (Dim.STATIC, Dim.STATIC, seq_len, Dim.STATIC)

shapes[attention_mask] = (Dim.STATIC, Dim.AUTO)
shapes[position_ids] = (Dim.STATIC, Dim.AUTO)
shapes[input_embeds] = (Dim.STATIC, Dim.AUTO, Dim.STATIC)
shapes[cache_position] = (Dim.AUTO,)

sample_kwargs["past_key_values"] = past_key_values
# need to use non-strict post-dispatch export to workaround some vmap issues in sdpa
# see https://github.com/pytorch/pytorch/issues/157323
ep = _export(
    mod=language_model, 
    args=(), 
    kwargs=sample_kwargs, 
    dynamic_shapes=shapes, # pyre-ignore
    pre_dispatch=False,
    strict=False
)
epm = ep.module()
ep_out = epm(**sample_kwargs)
torch.allclose(eager_out.logits, ep_out.logits)

# verify dynamic shape works
attention_mask_1 = torch.randint(low=0, high=2, size=(1, 1456), dtype=torch.int64, device=device)
position_ids_1 = torch.arange(0, 1456, dtype=torch.int64, device=device).unsqueeze(0)
input_embeds_1 = torch.rand((1, 1456, 3072), dtype=torch.bfloat16, device=device)
cache_position_1 = torch.arange(0, 1456, dtype=torch.int64, device=device)

sample_kwargs_1 = {
    "input_ids": None, 
    "attention_mask": attention_mask_1, 
    "position_ids": position_ids_1,
    "past_key_values": DynamicCache(),
    "inputs_embeds": input_embeds_1,
    "labels": None,
    "use_cache": True, 
    "cache_position": cache_position_1,
    "logits_to_keep": 1
}

eager_out_1 = language_model(**sample_kwargs_1)
sample_kwargs_1["past_key_values"] = eager_out_1.past_key_values
ep_out_1 = epm(**sample_kwargs_1)
torch.allclose(eager_out_1.logits, ep_out_1.logits)


#### export the embedding layer in the language model ####

embedding_layer = language_model.model.embed_tokens
# manually create example inputs
sample_inputs = torch.randint(low=0, high=16024, size=(1, 1138), dtype=torch.int64, device=device)
eager_out = embedding_layer(sample_inputs)

shapes = ShapesCollection()
shapes[sample_inputs] = (Dim.STATIC, Dim.AUTO)
ep = torch.export.export(mod=embedding_layer, args=(sample_inputs,), dynamic_shapes=shapes)
epm = ep.module()
ep_out = epm(sample_inputs)
torch.allclose(eager_out, ep_out)

# verify dynamic shape works
sample_inputs_1 = torch.randint(low=0, high=16024, size=(1, 1234), dtype=torch.int64, device=device)
eager_out_1 = embedding_layer(sample_inputs_1)
ep_out_1 = epm(sample_inputs_1)
torch.allclose(eager_out_1, ep_out_1)
