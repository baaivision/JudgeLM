#!/bin/bash

# w/o reference & w/o reverse
python ./judgelm/llm_judge/gen_model_judgement.py \
--model-path "/home/zhulianghui/ProjectC_ChatGPT/alpaca-quan/output/vicuna-7b-v1.3-data(judgelm-train-0628-gpt4-100k-w-reference-all-w-reference-drop)-bs128-ep3-lr2e-5-wd0.-wr0.03-cosine-mmlength2048-lazy-preprocess-swap-aug-ref_drop_ratio0.5" \
--model-id 7b-full-model-pycharm-debug \
--question-file ./judgelm/data/JudgeLM/judgelm_val_5k.jsonl \
--answer-file /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgements_output/JudgeLM/7b-full-model-pycharm-debug \
--num-gpus-per-model 1 \
--num-gpus-total 8 \
--temperature 0.2 \
--if-reverse 0 \
--if-fast-eval 1

# w/o reference & w/ reverse
python ./judgelm/llm_judge/gen_model_judgement.py \
--model-path "/home/zhulianghui/ProjectC_ChatGPT/alpaca-quan/output/vicuna-7b-v1.3-data(judgelm-train-0628-gpt4-100k-w-reference-all-w-reference-drop)-bs128-ep3-lr2e-5-wd0.-wr0.03-cosine-mmlength2048-lazy-preprocess-swap-aug-ref_drop_ratio0.5" \
--model-id 7b-full-model-pycharm-debug \
--question-file ./judgelm/data/JudgeLM/judgelm_val_5k.jsonl \
--answer-file /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgements_output/JudgeLM/7b-full-model-pycharm-debug-reverse \
--num-gpus-per-model 1 \
--num-gpus-total 8 \
--temperature 0.2 \
--if-reverse 1 \
--if-fast-eval 1

# w/ reference & w/o reverse
python ./judgelm/llm_judge/gen_model_judgement.py \
--model-path "/home/zhulianghui/ProjectC_ChatGPT/alpaca-quan/output/vicuna-7b-v1.3-data(judgelm-train-0628-gpt4-100k-w-reference-all-w-reference-drop)-bs128-ep3-lr2e-5-wd0.-wr0.03-cosine-mmlength2048-lazy-preprocess-swap-aug-ref_drop_ratio0.5" \
--model-id 7b-full-model-pycharm-debug \
--question-file ./judgelm/data/JudgeLM/judgelm_val_5k.jsonl \
--answer-file /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgements_output/JudgeLM/7b-full-model-pycharm-debug-w-ref \
--num-gpus-per-model 1 \
--num-gpus-total 8 \
--temperature 0.2 \
--reference-file /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgelm/data/JudgeLM/judgelm_val_5k_references.jsonl \
--if-reverse 0 \
--if-fast-eval 1

# w/ reference & w/ reverse
python ./judgelm/llm_judge/gen_model_judgement.py \
--model-path "/home/zhulianghui/ProjectC_ChatGPT/alpaca-quan/output/vicuna-7b-v1.3-data(judgelm-train-0628-gpt4-100k-w-reference-all-w-reference-drop)-bs128-ep3-lr2e-5-wd0.-wr0.03-cosine-mmlength2048-lazy-preprocess-swap-aug-ref_drop_ratio0.5" \
--model-id 7b-full-model-pycharm-debug \
--question-file ./judgelm/data/JudgeLM/judgelm_val_5k.jsonl \
--answer-file /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgements_output/JudgeLM/7b-full-model-pycharm-debug-w-ref-reverse \
--num-gpus-per-model 1 \
--num-gpus-total 8 \
--temperature 0.2 \
--reference-file /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgelm/data/JudgeLM/judgelm_val_5k_references.jsonl \
--if-reverse 1 \
--if-fast-eval 1

# Eval metrics w/o reference
python ./judgelm/llm_judge/eval_model_judgement.py \
--gt-answer-file-path /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgelm/data/JudgeLM/judgelm_val_5k_gpt4.jsonl \
--sequential-pred-answer-file-path /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgements_output/JudgeLM/7b-full-model-pycharm-debug \
--reversed-pred-answer-file-path /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgements_output/JudgeLM/7b-full-model-pycharm-debug-reverse

# Eval metrics w/ reference
python ./judgelm/llm_judge/eval_model_judgement.py \
--gt-answer-file-path /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgelm/data/JudgeLM/judgelm_val_5k_gpt4_with_reference.jsonl \
--sequential-pred-answer-file-path /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgements_output/JudgeLM/7b-full-model-pycharm-debug-w-ref \
--reversed-pred-answer-file-path /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgements_output/JudgeLM/7b-full-model-pycharm-debug-w-ref-reverse