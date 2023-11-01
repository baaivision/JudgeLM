#!/bin/bash

# mv
# mv /Projects/Vicuna/predictions/results-23-10-10.jsonl ./judgelm/data/JudgeLM/answers/vicuna_judgelm_val.json
# mv /Projects/LLaMA/predictions/results-23-10-10.jsonl ./judgelm/data/JudgeLM/answers/llama_judgelm_val.json

# preprocess
python ./judgelm/data/JudgeLM/judgelm_preprocess.py \
--ans1_file_path /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgelm/data/JudgeLM/answers/gt_judgelm_val.jsonl \
--ans2_file_path /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgelm/data/JudgeLM/answers/llama_judgelm_val.jsonl

# judge
python ./judgelm/llm_judge/gen_model_judgement_single.py \
--model-path "/home/zhulianghui/ProjectC_ChatGPT/alpaca-quan/output/vicuna-7b-v1.3-data(judgelm-train-0628-gpt4-100k-w-reference-all-w-reference-drop)-bs128-ep3-lr2e-5-wd0.-wr0.03-cosine-mmlength2048-lazy-preprocess-swap-aug-ref_drop_ratio0.5" \
--model-id 7b-full-model \
--question-file ./judgelm/data/JudgeLM/judgelm-val-5k-judge-samples.jsonl \
--answer-file /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgements_output/JudgeLM/7b-full-model-single-ans \
--num-gpus-per-model 1 \
--num-gpus-total 8 \
--temperature 0.2 \
--reference-file /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgelm/data/JudgeLM/judgelm_val_5k_references.jsonl \
--if-fast-eval 1