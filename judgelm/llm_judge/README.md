# JudgeLM for different benchmarks

JudgeLM presents a strong 0-shot ability to many open-ended benchmarks.
Now we support the following benchmarks:
- JudgeLM val set
- MM-Vet

For simplicity of use, you can download our uploaded [dataset collection](https://huggingface.co/datasets/BAAI/JudgeLM-data-collection-v1.0),
and put the contents in the `JudgeLM/judgelm/data` folder.

[//]: # (## Enviorment)

[//]: # ()
[//]: # (```)

[//]: # (/share/project/lianghuizhu/JudgeLM-Project/judgelm-conda-env/bin/python)

[//]: # (```)

[//]: # ()
[//]: # (## Optional Checkpoints)

[//]: # (```)

[//]: # (JudgeLM-7B-full-model: /share/project/lianghuizhu/JudgeLM-Project/judgelm-7b-v1.0-full-model)

[//]: # (JudgeLM-13B-full-model: /share/project/lianghuizhu/JudgeLM-Project/judgelm-13b-v1.0-full-model)

[//]: # (JudgeLM-33B-full-model: /share/project/lianghuizhu/JudgeLM-Project/judgelm-33b-v1.0-full-model)

[//]: # (```)

## Judge the Quality of LLMs-Generated Answer Pairs

We provide scripts to judge the quality of LLMs-generated answer pairs. We first put the LLM-generated results json file into benchmark folder, and then preprocess for judge samples. Finally, we make judgements by JudgeLM. Furthermore, we provide a single script to run the whole process at last.

### Judge on JudgeLM Benchmark Step by Step

#### Step 1. Put LLM-generated results json file in benchmark folder `./judgelm/data/JudgeLM/answers`, following the format of `./judgelm/data/JudgeLM/answers/alpaca_judgelm_val.jsonl`

e.g.,

```json
{
  "question_id": 0, 
  "question_body": "My internet service is very expensive...", 
  "decoding_method": "top_p_sampling", 
  "model": "alpaca-native", 
  "text": "There are a few ways to cut down the cost...", 
  "scores": {"logprobs": -7.0179795026779175,...}
}
```

Among the keys, `question_id`, `question_body`, `text` are required, and `decoding_method`, `model`, `scores` are optional (can be some placeholder).

#### Step 2. Preprocess for judge samples


```bash
python ./judgelm/data/JudgeLM/judgelm_preprocess.py \
--ans1_file_path [ANS1_FILE_PATH] \
--ans2_file_path [ANS2_FILE_PATH]
```

Arguments:
  - `[ANS1_FILE_PATH]` is the path to the answer file 1.
  - `[ANS2_FILE_PATH]` is the path to the answer file 2.

e.g.,

```bash
python ./judgelm/data/JudgeLM/judgelm_preprocess.py \
--ans1_file_path /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgelm/data/JudgeLM/answers/vicuna_judgelm_val.jsonl \
--ans2_file_path /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgelm/data/JudgeLM/answers/llama_judgelm_val.jsonl
```

After this, we can get judge samples like `./judgelm/data/JudgeLM/judgelm-val-5k-judge-samples.jsonl`

#### Step 3. Make judgements by JudgeLM

```bash
python ./judgelm/llm_judge/gen_model_judgement.py \ 
--model-path [MODEL_PATH] \
--model-id [MODEL_ID] \
--question-file ./judgelm/data/JudgeLM/judgelm-val-5k-judge-samples.jsonl \
--answer-file [ANSWER_FILE_PATH] \
--num-gpus-per-model [NUM_GPUS_PER_MODEL] \
--num-gpus-total [NUM_GPUS_TOTAL] \
--temperature [TEMPERATURE] \
--reference-file [REFERENCE_FILE_PATH] \
--if-fast-eval [IF_FAST_EVAL] 
```

Arguments:
  - `[MODEL_PATH]` is the path to the judge weights, which can be a local folder.
  - `[MODEL_ID]` is a name you give to the judge model.
  - `[ANSWER_FILE_PATH]` is the path to the output judgements.
  - `[IF_FAST_EVAL]`, int, 0 or 1, represents if use the fast eval (without reasons generation).
  - `[ANSWER_FILE_PATH]` is the path to the output judgements.
  - `[NUM_GPUS_PER_MODEL]` is the gpu nums that used to run the judge model.
  - `[NUM_GPUS_TOTAL]` is the total gpu nums that used to run the judge model.
  - `[TEMPERATURE]` is the temperature used to sample the judge model.
  - `[REFERENCE_FILE_PATH]`, is the path to the reference answers. It can be "None" if you dont need the JudgeLM make judgements with reference answers.
  - `[IF_FAST_EVAL]`, int, 0 or 1, represents if use the fast eval (without reasons generation).

e.g.,

```bash
python ./judgelm/llm_judge/gen_model_judgement.py \
--model-path "/home/zhulianghui/ProjectC_ChatGPT/alpaca-quan/output/vicuna-7b-v1.3-data(judgelm-train-0628-gpt4-100k-w-reference-all-w-reference-drop)-bs128-ep3-lr2e-5-wd0.-wr0.03-cosine-mmlength2048-lazy-preprocess-swap-aug-ref_drop_ratio0.5" \
--model-id 7b-full-model \
--question-file ./judgelm/data/JudgeLM/judgelm-val-5k-judge-samples.jsonl \
--answer-file /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgements_output/JudgeLM/7b-full-model \
--num-gpus-per-model 1 \
--num-gpus-total 8 \
--temperature 0.2 \
--reference-file /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgelm/data/JudgeLM/judgelm_val_5k_references.jsonl \
--if-fast-eval 1 
```

### ⭐️ Judge on JudgeLM Benchmark with a Single Script

```bash
bash ./scripts/judge_on_judgelm_benchmark.sh 
```

## Grade the Single Answer
Given a question and the corresponding reference answer, JudgeLM can grade the LLM-generated single answer. 
We first put the LLM-generated results json file into benchmark folder, and then preprocess the reference answer file and the LLM-generated answer file. 
Finally, we grade single answer by JudgeLM. Furthermore, we provide a single script to run the whole process at last.

### Grade Single Answer on JudgeLM Benchmark Step by Step

#### Step 1. Put LLM-generated results json file in benchmark folder `./judgelm/data/JudgeLM/answers`, following the format of `./judgelm/data/JudgeLM/answers/alpaca_judgelm_val.jsonl`

e.g.,

```json
{
  "question_id": 0, 
  "question_body": "My internet service is very expensive...", 
  "decoding_method": "top_p_sampling", 
  "model": "alpaca-native", 
  "text": "There are a few ways to cut down the cost...", 
  "scores": {"logprobs": -7.0179795026779175,...}
}
```

Among the keys, `question_id`, `question_body`, `text` are required, and `decoding_method`, `model`, `scores` are optional (can be some placeholder).

#### Step 2. Preprocess for judge samples


```bash
python ./judgelm/data/JudgeLM/judgelm_preprocess.py \
--ans1_file_path [REFERENCE_ANSWER_FILE_PATH] \
--ans2_file_path [ANS2_FILE_PATH]
```

Arguments:
  - `[REFERENCE_ANSWER_FILE_PATH]` is the path to the reference answer file.
  - `[ANS2_FILE_PATH]` is the path to the LLM-generated answer file.

e.g.,

```bash
python ./judgelm/data/JudgeLM/judgelm_preprocess.py \
--ans1_file_path /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgelm/data/JudgeLM/answers/gt_judgelm_val.jsonl \
--ans2_file_path /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgelm/data/JudgeLM/answers/llama_judgelm_val.jsonl
```

After this, we can get judge samples like `./judgelm/data/JudgeLM/judgelm-val-5k-judge-samples.jsonl`

#### Step 3. Make judgements by JudgeLM

```bash
python ./judgelm/llm_judge/gen_model_judgement_single.py \ 
--model-path [MODEL_PATH] \
--model-id [MODEL_ID] \
--question-file ./judgelm/data/JudgeLM/judgelm-val-5k-judge-samples.jsonl \
--answer-file [ANSWER_FILE_PATH] \
--num-gpus-per-model [NUM_GPUS_PER_MODEL] \
--num-gpus-total [NUM_GPUS_TOTAL] \
--temperature [TEMPERATURE] \
--reference-file [REFERENCE_FILE_PATH] \
--if-fast-eval [IF_FAST_EVAL] 
```

Arguments:
  - `[MODEL_PATH]` is the path to the judge weights, which can be a local folder.
  - `[MODEL_ID]` is a name you give to the judge model.
  - `[ANSWER_FILE_PATH]` is the path to the output judgements.
  - `[IF_FAST_EVAL]`, int, 0 or 1, represents if use the fast eval (without reasons generation).
  - `[ANSWER_FILE_PATH]` is the path to the output judgements.
  - `[NUM_GPUS_PER_MODEL]` is the gpu nums that used to run the judge model.
  - `[NUM_GPUS_TOTAL]` is the total gpu nums that used to run the judge model.
  - `[TEMPERATURE]` is the temperature used to sample the judge model.
  - `[REFERENCE_FILE_PATH]`, is the path to the reference answers. It can be "None" if you dont need the JudgeLM make judgements with reference answers.
  - `[IF_FAST_EVAL]`, int, 0 or 1, represents if use the fast eval (without reasons generation).

e.g.,

```bash
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
```

### ⭐️ Grade the Single Answer on JudgeLM Benchmark with a Single Script

```bash
bash ./scripts/grade_single_answer_on_judgelm_benchmark.sh 
```

## Grade Multiple Answers
JudgeLM also can grade multiple LLM-generated answers based on a given question.
We first put the LLM-generated results json file into benchmark folder, and then preprocess the LLM-generated answer files. 
Finally, we grade multiple answers by JudgeLM. Furthermore, we provide a single script to run the whole process at last.

Note, the number of multi-answers to be judged is limited by the JudgeLM's context length (2048).

### Grade Multiple Answers on JudgeLM Benchmark Step by Step

#### Step 1. Put LLM-generated results json file in benchmark folder `./judgelm/data/JudgeLM/answers`, following the format of `./judgelm/data/JudgeLM/answers/alpaca_judgelm_val.jsonl`

e.g.,

```json
{
  "question_id": 0, 
  "question_body": "My internet service is very expensive...", 
  "decoding_method": "top_p_sampling", 
  "model": "alpaca-native", 
  "text": "There are a few ways to cut down the cost...", 
  "scores": {"logprobs": -7.0179795026779175,...}
}
```

Among the keys, `question_id`, `question_body`, `text` are required, and `decoding_method`, `model`, `scores` are optional (can be some placeholder).

#### Step 2. Preprocess for judge samples


```bash
python ./judgelm/data/JudgeLM/judgelm_preprocess.py \
--ans1_file_path [ANS1_FILE_PATH] \
--ans2_file_path [ANS2_FILE_PATH] \
--ansmore_file_paths [ANSMORE_FILE_PATH]
```

Arguments:
  - `[ANS1_FILE_PATH]` is the path to the answer file 1.
  - `[ANS2_FILE_PATH]` is the path to the answer file 2.
  - `[ANSMORE_FILE_PATH]` is the paths to the extra answer files.

e.g.,

```bash
python ./judgelm/data/JudgeLM/judgelm_preprocess.py \
--ans1_file_path /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgelm/data/JudgeLM/answers/alpaca_judgelm_val.jsonl \
--ans2_file_path /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgelm/data/JudgeLM/answers/chatglm_judgelm_val.jsonl \
--ansmore_file_paths /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgelm/data/JudgeLM/answers/dolly_judgelm_val.jsonl /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgelm/data/JudgeLM/answers/flant5_judgelm_val.jsonl
```

After this, we can get judge samples like `./judgelm/data/JudgeLM/judgelm-val-5k-judge-samples.jsonl`

#### Step 3. Make judgements by JudgeLM

```bash
python ./judgelm/llm_judge/gen_model_judgement_multi.py \ 
--model-path [MODEL_PATH] \
--model-id [MODEL_ID] \
--question-file ./judgelm/data/JudgeLM/judgelm-val-5k-judge-samples.jsonl \
--answer-file [ANSWER_FILE_PATH] \
--num-gpus-per-model [NUM_GPUS_PER_MODEL] \
--num-gpus-total [NUM_GPUS_TOTAL] \
--temperature [TEMPERATURE] \
--reference-file [REFERENCE_FILE_PATH] \
--if-fast-eval [IF_FAST_EVAL] \
--answer-num [ANSWER_NUM]
```

Arguments:
  - `[MODEL_PATH]` is the path to the judge weights, which can be a local folder.
  - `[MODEL_ID]` is a name you give to the judge model.
  - `[ANSWER_FILE_PATH]` is the path to the output judgements.
  - `[IF_FAST_EVAL]`, int, 0 or 1, represents if use the fast eval (without reasons generation).
  - `[ANSWER_FILE_PATH]` is the path to the output judgements.
  - `[NUM_GPUS_PER_MODEL]` is the gpu nums that used to run the judge model.
  - `[NUM_GPUS_TOTAL]` is the total gpu nums that used to run the judge model.
  - `[TEMPERATURE]` is the temperature used to sample the judge model.
  - `[REFERENCE_FILE_PATH]`, is the path to the reference answers. It can be "None" if you dont need the JudgeLM make judgements with reference answers.
  - `[IF_FAST_EVAL]`, int, 0 or 1, represents if use the fast eval (without reasons generation).
  - `[ANSWER_NUM]`, int, larger than 2, is the number of answers to grade.

e.g.,

```bash
python ./judgelm/llm_judge/gen_model_judgement_multi.py \
--model-path "/home/zhulianghui/ProjectC_ChatGPT/alpaca-quan/output/vicuna-7b-v1.3-data(judgelm-train-0628-gpt4-100k-w-reference-all-w-reference-drop)-bs128-ep3-lr2e-5-wd0.-wr0.03-cosine-mmlength2048-lazy-preprocess-swap-aug-ref_drop_ratio0.5" \
--model-id 7b-full-model \
--question-file ./judgelm/data/JudgeLM/judgelm-val-5k-judge-samples.jsonl \
--answer-file /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgements_output/JudgeLM/7b-full-model-multi-ans \
--num-gpus-per-model 1 \
--num-gpus-total 8 \
--temperature 0.2 \
--reference-file /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgelm/data/JudgeLM/judgelm_val_5k_references.jsonl \
--if-fast-eval 1 \
--answer-num 4
```

### ⭐️ Grade Multiple Answers on JudgeLM Benchmark with a Single Script

```bash
bash ./scripts/grade_multi_answer_on_judgelm_benchmark.sh 
```


## Evaluating the Performence of Judge Models

We also provide scripts to evaluate the performence of judge models. We first generate the judgements in different situations, and then calculate the metrics of JudgeLM. Furthermore, we provide a single script to run the whole process at last.
### Evaluate Judge on JudgeLM Benchmark Step by Step

We provide scripts to evaluate the judge's performance on JudgeLM val set. We first generate the judgements in situation of `w/o reference & w/o reverse `, `w/o reference & w/ reverse `, `w/ reference & w/o reverse `, `w/ reference & w/ reverse `, and then calculate the metrics of JudgeLM.

#### Step 1. Make judgements by JudgeLM

```bash
python ./judgelm/llm_judge/gen_model_judgement.py \
--model-path [MODEL_PATH] \
--model-id [MODEL_ID] \
--question-file ./judgelm/data/JudgeLM/judgelm_val_5k.jsonl \
--answer-file [ANSWER_FILE_PATH] \
--num-gpus-per-model [NUM_GPUS_PER_MODEL] \
--num-gpus-total [NUM_GPUS_TOTAL] \
--temperature [TEMPERATURE] \
--if-reverse [IF_REVERSE] \
--if-fast-eval [IF_FAST_EVAL]
```

Arguments:
  - `[MODEL_PATH]` is the path to the judge weights, which can be a local folder.
  - `[MODEL_ID]` is a name you give to the judge model.
  - `[ANSWER_FILE_PATH]` is the path to the output judgements.
  - `[NUM_GPUS_PER_MODEL]` is the gpu nums that used to run the judge model.
  - `[NUM_GPUS_TOTAL]` is the total gpu nums that used to run the judge model.
  - `[TEMPERATURE]` is the temperature used to sample the judge model.
  - `[IF_REVERSE]`, int, 0 or 1, represents if reverse the answer1 and answer2.
  - `[IF_FAST_EVAL]`, int, 0 or 1, represents if use the fast eval (without reasons generation).

e.g.,

```bash
# w/o reference & w/o reverse
python ./judgelm/llm_judge/gen_model_judgement.py \
--model-path "/share/project/lianghuizhu/JudgeLM-Project/judgelm-7b-v1.0-full-model" \
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
--model-path "/share/project/lianghuizhu/JudgeLM-Project/judgelm-7b-v1.0-full-model" \
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
--model-path "/share/project/lianghuizhu/JudgeLM-Project/judgelm-7b-v1.0-full-model" \
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
--model-path "/share/project/lianghuizhu/JudgeLM-Project/judgelm-7b-v1.0-full-model" \
--model-id 7b-full-model-pycharm-debug \
--question-file ./judgelm/data/JudgeLM/judgelm_val_5k.jsonl \
--answer-file /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgements_output/JudgeLM/7b-full-model-pycharm-debug-w-ref-reverse \
--num-gpus-per-model 1 \
--num-gpus-total 8 \
--temperature 0.2 \
--reference-file /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgelm/data/JudgeLM/judgelm_val_5k_references.jsonl \
--if-reverse 1 \
--if-fast-eval 1 
```

#### Step 2. Calculate the metrics of JudgeLM

```bash
python ./judgelm/llm_judge/eval_model_judgement.py \
--gt-answer-file-path [GT_ANSWER_FILE_PATH] \
--sequential-pred-answer-file-path [SEQUENTIAL_PRED_ANSWER_FILE_PATH] \
--reversed-pred-answer-file-path [REVERSED_PRED_ANSWER_FILE_PATH] 
```

Arguments:
  - `[GT_ANSWER_FILE_PATH]` is the path to the ground truth answers.
  - `[SEQUENTIAL_PRED_ANSWER_FILE_PATH]` is the path to the sequential predicted answers.
  - `[REVERSED_PRED_ANSWER_FILE_PATH]` is the path to the reversed predicted answers.

e.g.,

```bash
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
```

### ⭐️ Evaluate Judge on JudgeLM Benchmark with a Single Script

```bash
bash ./scripts/eval_judge_on_judgelm_benchmark.sh 
```

## MM-Vet Benchmark

The proposed JudgeLM is easy to apply in modern multimodal benchmarks, e.g., MM-Vet. First, we put the multimodal results json file in benchmark folder, and then preprocess for judge samples. Finally, we make judgements by JudgeLM. Furthermore, we provide a single script to run the whole process at last.

### Evaluate on MM-Vet Benchmark through JudgeLM Step by Step

#### Step 1. Put Multimodal-results json file in benchmark folder

```bash
mv /path/to/reuslts.json ./judgelm/data/MM-Vet/mm-vet-xxx-prediction.json
```

e.g.,

```bash
mv /Projects/Emu/predictions/results-23-10-10.json ./judgelm/data/MM-Vet/mm-vet-emu-prediction.json
```

#### Step 2. Preprocess for judge samples

```bash
python ./judgelm/data/MM-Vet/mmvet_preprocess.py --gt_file_path ./judgelm/data/MM-Vet/mm-vet-gt.json --pred_file_path ./judgelm/data/MM-Vet/mm-vet-xxx-prediction.json
```

e.g.,
```bash
python ./judgelm/data/MM-Vet/mmvet_preprocess.py --gt_file_path ./judgelm/data/MM-Vet/mm-vet-gt.json --pred_file_path ./judgelm/data/MM-Vet/mm-vet-emu-prediction.json
```

After this, we can get judge samples like `./judgelm/data/MM-Vet/mm-vet-judge-samples.jsonl`

#### Step 3. Make judgements by JudgeLM

```bash
python ./judgelm/llm_judge/gen_model_judgement_mmvet.py --model-path [MODEL_PATH] --model-id [MODEL_ID] --question-file ./judgelm/data/MM-Vet/mmvet_predictions.jsonl --answer-file [ANSWER_FILE_PATH] --num-gpus-per-model [NUM_GPUS_PER_MODEL] --num-gpus-total [NUM_GPUS_TOTAL] --temperature [TEMPERATURE] --if-fast-eval [IF_FAST_EVAL] 
```

Arguments:
  - `[MODEL_PATH]` is the path to the judge weights, which can be a local folder.
  - `[MODEL_ID]` is a name you give to the judge model.
  - `[ANSWER_FILE_PATH]` is the path to the output judgements.
  - `[IF_FAST_EVAL]`, int, 0 or 1, represents if use the fast eval (without reasons generation).

e.g.,
```bash
python ./judgelm/llm_judge/gen_model_judgement_mmvet.py --model-path ./checkpoints_output/judgelm-33b-v1.0-full-model --model-id 33b-full-model --question-file ./judgelm/data/MM-Vet/mmvet_predictions.jsonl --answer-file ./judgements_output/MM-Vet/33b-full-model --num-gpus-per-model 2 --num-gpus-total 4 --temperature 0.2 --if-fast-eval 1 
```

### ⭐️Evaluate on MM-Vet Benchmark with a Single Script

```bash
bash ./scripts/judge_on_mmvet_benchmark.sh 
```