from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator
from transformers import AutoModelForSeq2SeqLM,AutoModelForCausalLM
from peft import IA3Config, get_peft_model,TaskType,BEFTConfig
import torch
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm


from transformers import AutoModelForSeq2SeqLM
from peft import BEFTModel, BEFTConfig

# config = BEFTConfig(
#      peft_type="BEFT",
#      task_type="SEQ_2_SEQ_LM",
#      target_modules=["v"],
#      )

# model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")#("t5-base")
# beft_model = BEFTModel(model,config,adapter_name="default")

# MT5ForConditionalGeneration(
#   (shared): Embedding(250112, 512)
#   (encoder): MT5Stack(
#     (embed_tokens): Embedding(250112, 512)
#     (block): ModuleList(
#       (0): MT5Block(
#         (layer): ModuleList(
#           (0): MT5LayerSelfAttention(
#             (SelfAttention): MT5Attention(
#               (q): Linear(in_features=512, out_features=384, bias=False)
#               (k): Linear(in_features=512, out_features=384, bias=False)
#               (v): Linear(in_features=512, out_features=384, bias=False)
#               (o): Linear(in_features=384, out_features=512, bias=False)
#               (relative_attention_bias): Embedding(32, 6)
#             )
#             (layer_norm): MT5LayerNorm()
#             (dropout): Dropout(p=0.1, inplace=False)
#           )
#           (1): MT5LayerFF(
#             (DenseReluDense): MT5DenseGatedActDense(
#               (wi_0): Linear(in_features=512, out_features=1024, bias=False)
#               (wi_1): Linear(in_features=512, out_features=1024, bias=False)
#               (wo): Linear(in_features=1024, out_features=512, bias=False)
#               (dropout): Dropout(p=0.1, inplace=False)
#               (act): NewGELUActivation()
#             )
#             (layer_norm): MT5LayerNorm()
#             (dropout): Dropout(p=0.1, inplace=False)
#           )
#         )
#       )
#       (1-7): 7 x MT5Block(
#         (layer): ModuleList(
#           (0): MT5LayerSelfAttention(
#             (SelfAttention): MT5Attention(
#               (q): Linear(in_features=512, out_features=384, bias=False)
#               (k): Linear(in_features=512, out_features=384, bias=False)
#               (v): Linear(in_features=512, out_features=384, bias=False)
#               (o): Linear(in_features=384, out_features=512, bias=False)
#             )
#             (layer_norm): MT5LayerNorm()
#             (dropout): Dropout(p=0.1, inplace=False)
#           )
#           (1): MT5LayerFF(
#             (DenseReluDense): MT5DenseGatedActDense(
#               (wi_0): Linear(in_features=512, out_features=1024, bias=False)
#               (wi_1): Linear(in_features=512, out_features=1024, bias=False)
#               (wo): Linear(in_features=1024, out_features=512, bias=False)
#               (dropout): Dropout(p=0.1, inplace=False)
#               (act): NewGELUActivation()
#             )
#             (layer_norm): MT5LayerNorm()
#             (dropout): Dropout(p=0.1, inplace=False)
#           )
#         )
#       )
#     )
#     (final_layer_norm): MT5LayerNorm()
#     (dropout): Dropout(p=0.1, inplace=False)
#   )
#   (decoder): MT5Stack(
#     (embed_tokens): Embedding(250112, 512)
#     (block): ModuleList(
#       (0): MT5Block(
#         (layer): ModuleList(
#           (0): MT5LayerSelfAttention(
#             (SelfAttention): MT5Attention(
#               (q): Linear(in_features=512, out_features=384, bias=False)
#               (k): Linear(in_features=512, out_features=384, bias=False)
#               (v): Linear(in_features=512, out_features=384, bias=False)
#               (o): Linear(in_features=384, out_features=512, bias=False)
#               (relative_attention_bias): Embedding(32, 6)
#             )
#             (layer_norm): MT5LayerNorm()
#             (dropout): Dropout(p=0.1, inplace=False)
#           )
#           (1): MT5LayerCrossAttention(
#             (EncDecAttention): MT5Attention(
#               (q): Linear(in_features=512, out_features=384, bias=False)
#               (k): Linear(in_features=512, out_features=384, bias=False)
#               (v): Linear(in_features=512, out_features=384, bias=False)
#               (o): Linear(in_features=384, out_features=512, bias=False)
#             )
#             (layer_norm): MT5LayerNorm()
#             (dropout): Dropout(p=0.1, inplace=False)
#           )
#           (2): MT5LayerFF(
#             (DenseReluDense): MT5DenseGatedActDense(
#               (wi_0): Linear(in_features=512, out_features=1024, bias=False)
#               (wi_1): Linear(in_features=512, out_features=1024, bias=False)
#               (wo): Linear(in_features=1024, out_features=512, bias=False)
#               (dropout): Dropout(p=0.1, inplace=False)
#               (act): NewGELUActivation()
#             )
#             (layer_norm): MT5LayerNorm()
#             (dropout): Dropout(p=0.1, inplace=False)
#           )
#         )
#       )
#       (1-7): 7 x MT5Block(
#         (layer): ModuleList(
#           (0): MT5LayerSelfAttention(
#             (SelfAttention): MT5Attention(
#               (q): Linear(in_features=512, out_features=384, bias=False)
#               (k): Linear(in_features=512, out_features=384, bias=False)
#               (v): Linear(in_features=512, out_features=384, bias=False)
#               (o): Linear(in_features=384, out_features=512, bias=False)
#             )
#             (layer_norm): MT5LayerNorm()
#             (dropout): Dropout(p=0.1, inplace=False)
#           )
#           (1): MT5LayerCrossAttention(
#             (EncDecAttention): MT5Attention(
#               (q): Linear(in_features=512, out_features=384, bias=False)
#               (k): Linear(in_features=512, out_features=384, bias=False)
#               (v): Linear(in_features=512, out_features=384, bias=False)
#               (o): Linear(in_features=384, out_features=512, bias=False)
#             )
#             (layer_norm): MT5LayerNorm()
#             (dropout): Dropout(p=0.1, inplace=False)
#           )
#           (2): MT5LayerFF(
#             (DenseReluDense): MT5DenseGatedActDense(
#               (wi_0): Linear(in_features=512, out_features=1024, bias=False)
#               (wi_1): Linear(in_features=512, out_features=1024, bias=False)
#               (wo): Linear(in_features=1024, out_features=512, bias=False)
#               (dropout): Dropout(p=0.1, inplace=False)
#               (act): NewGELUActivation()
#             )
#             (layer_norm): MT5LayerNorm()
#             (dropout): Dropout(p=0.1, inplace=False)
#           )
#         )
#       )
#     )
#     (final_layer_norm): MT5LayerNorm()
#     (dropout): Dropout(p=0.1, inplace=False)
#   )
#   (lm_head): Linear(in_features=512, out_features=250112, bias=False)
# )

ds = load_dataset("financial_phrasebank", "sentences_allagree",trust_remote_code=True)
ds = ds["train"].train_test_split(test_size=0.1)
ds["validation"] = ds["test"]
del ds["test"]

classes = ds["train"].features["label"].names
ds = ds.map(
    lambda x: {"text_label": [classes[label] for label in x["label"]]},
    batched=True,
    num_proc=1,
)

print(ds["train"][0])
# {'sentence': 'It will be operated by Nokia , and supported by its Nokia NetAct network and service management system .',
#  'label': 1,
#  'text_label': 'neutral'}


text_column = "sentence"
label_column = "text_label"
max_length = 128

tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-small")#("google/mt5-small")
#tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")


def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[label_column]
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=3, padding="max_length", truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

processed_ds = ds.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=ds["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_ds = processed_ds["train"].select(range(500))
eval_ds = processed_ds["validation"]

batch_size = 8

train_dataloader = DataLoader(
    train_ds, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_ds, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

#model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-small")#("google/mt5-small")
#print(model)

peft_config = BEFTConfig(task_type="SEQ_2_SEQ_LM",target_modules=["v"])#,target_modules=["v"]) #"all-linear"
#peft_config = IA3Config(task_type=TaskType.CAUSAL_LM,inference_mode=False)
model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())

lr = 8e-3
num_epochs = 1

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)



#device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")