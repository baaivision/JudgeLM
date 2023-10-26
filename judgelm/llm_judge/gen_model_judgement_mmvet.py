"""Generate answers with local models.

"""
import argparse
import json
import os
import time

import shortuuid
import torch
from tqdm import tqdm

import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
root = file.parents[2]
sys.path.append(str(root))
print(sys.path)

from judgelm.llm_judge.common import load_questions, reorg_answer_file, conv_judge_vqa_single_answer, KeywordsStoppingCriteria
from judgelm.model import load_model


def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    temperature,
    if_fast_eval
):
    questions = load_questions(question_file, question_begin, question_end)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model) # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                if_fast_eval,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_gpus_per_model,
    max_gpu_memory,
    temperature,
    if_fast_eval,
):
    model, tokenizer = load_model(
        model_path,
        device="cuda",
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )

    for q_i, question in tqdm(enumerate(questions)):
        torch.manual_seed(q_i)
        conv = conv_judge_vqa_single_answer.copy()
        template = conv.prompt_template

        # if fast eval, use the "\n" as the separator
        if if_fast_eval:
            conv.sep = "\n"

        # preprocess reference answer
        if "<OR>" in question['answer']:
            answers = question['answer'].split('<OR>')
            import random
            placeholder_answer = random.choice(answers)
        elif "<AND>" in question['answer']:
            placeholder_answer = question['answer'].replace("<AND>", " and ")
        else:
            placeholder_answer = question['answer']

        # combine data_sample
        data_sample = conv.system + '\n' + template.format(question=question['question'],
                                                                            reference=question['answer'],
                                                                            answer_1=placeholder_answer,
                                                                            answer_2=question['answer1_body'],
                                                                            prompt=conv.prompt) + conv.appendix

        input_ids = tokenizer([data_sample]).input_ids
        input_ids[0][0] = 1

        do_sample = False if temperature < 1e-4 else True
        stopping_criteria = KeywordsStoppingCriteria([conv.sep], tokenizer, torch.as_tensor(input_ids))

        # generate judgements
        output_ids = model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_token,
            stopping_criteria=[stopping_criteria]  
        )

        if model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]) :]
        
        output = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
        )

        if conv.sep:
            output = output[: output.find(conv.sep)]
        output = output.strip()

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_id = shortuuid.uuid()
            question["pred_id"] = ans_id
            question["pred_text"] = output
            question["pred_model_id"] = model_id
            question["tstamp"] = time.time()
            fout.write(json.dumps(question) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--question-file",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=2048,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        # default="37Gib",
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--if-fast-eval",
        type=int,
        default=0,
        help="Whether to use fast evaluation.",
    )
    args = parser.parse_args()
    args.if_fast_eval = bool(args.if_fast_eval)
    print(f"args: {args}")

    # if use ray
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray
        # ray.init(num_cpus=int(args.num_gpus_total / args.num_gpus_per_model))
        ray.init(num_cpus=int(args.num_gpus_total / args.num_gpus_per_model), runtime_env={"working_dir": str(root)})

    print(f"Output to {args.answer_file}")

    run_eval(
        args.model_path,
        args.model_id,
        args.question_file,
        args.question_begin,
        args.question_end,
        args.answer_file,
        args.max_new_token,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args.if_fast_eval
    )

    reorg_answer_file(args.answer_file)