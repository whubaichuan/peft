# BEFT: Bias-Efficient Fine-Tuning of Language Models in Low-Data Regimes

## Introduction
Fine-tuning the bias terms of large language models (LLMs) has the potential to achieve unprecedented parameter efficiency while maintaining competitive performance, particularly **in low-data regimes**. In this paper, we investigate the link between fine-tuning **b**<sub>q</sub>, **b**<sub>k</sub>, and **b**<sub>v</sub> with the performance of the downstream task, both analytically and empirically. We study and shed light on the expressive power of bias terms **b**<sub>q</sub>, **b**<sub>k</sub>, and **b**<sub>v</sub> in the query, key, or value projections of LLMs including bias-term-free LLMs. Our key finding is that directly fine-tuning **b**<sub>v</sub> generally leads to higher downstream performance in low-data regimes, in comparison to **b**<sub>q</sub> and **b**<sub>k</sub>.



## Quick start
You can try target_modules=`["v"]`, or `["q"]`, or `["k"]` in `beft_finetuning.py` to see the downstream accuracy.


## Current Support Models and The Default Target Modules

```python
TRANSFORMERS_MODELS_TO_BEFT_TARGET_MODULES_MAPPING = {
    "t5": ["v"],
    "mt5": ["v"],
    "roberta": ["value"],
    "opt": ["v_proj"],
    "gptj": ["v_proj"],
    "gpt_neo": ["v_proj"],
    "bart": ["v_proj"],
    "llama": ["v_proj"],
    "llama4": ["v_proj"],
    "mistral": ["v_proj"],
    "mixtral": ["v_proj"],
    "bert": ["value"],
    "deberta-v2": ["value_proj"],
    "phi": ["v_proj"],
    "gemma": ["v_proj"],
    "gemma2": ["v_proj"],
    "gemma3_text": ["v_proj"],
    "qwen2": ["v_proj"],
    "qwen3": ["v_proj"],
}
```

## Citation
```bibtex
@inproceedings{huang2026beft,
  title={BEFT: Bias-Efficient Fine-Tuning of Language Models in Low-Data Regimes},
  author={Huang, Baichuan and Balashankar, Ananth and Aminifar, Amir},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
  year={2026}
}
```
