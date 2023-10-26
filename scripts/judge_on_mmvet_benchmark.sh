#!/bin/bash

# mv
# mv /Projects/Emu/predictions/results-23-10-10.json ./judgelm/data/MM-Vet/mm-vet-emu-prediction.json 

# preprocess
python ./judgelm/data/MM-Vet/mmvet_preprocess.py --gt_file_path ./judgelm/data/MM-Vet/mm-vet-gt.json --pred_file_path ./judgelm/data/MM-Vet/mm-vet-emu-prediction.json

# judge
python ./judgelm/llm_judge/gen_model_judgement_mmvet.py --model-path /share/project/lianghuizhu/JudgeLM-Project/judgelm-33b-v1.0-full-model --model-id 33b-full-model --question-file ./judgelm/data/MM-Vet/mm-vet-judge-samples.jsonl --answer-file ./judgements_output/MM-Vet/judgelm-33b-v1.0-full-model-judgements --num-gpus-per-model 2 --num-gpus-total 4 --temperature 0.2 --if-fast-eval 1 
