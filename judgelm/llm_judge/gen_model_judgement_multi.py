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

from judgelm.llm_judge.common import load_questions, reorg_answer_file, conv_judge_pair, conv_judge_pair_w_reference, KeywordsStoppingCriteria, parse_score, translate_score_to_win_list
from judgelm.model import load_model
from judgelm.utils import extract_jsonl

# create num to words dict
num2words = {1:"one", 2:"two", 3:"three", 4:"four", 5:"five",
             6:"six", 7:"seven", 8: "eight", 9: 'nine', 10: 'ten', \
            11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', \
            15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen'}


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
    if_reverse_answers,
    reference_file,
    if_fast_eval,
    answer_num
):
    print("start run_eval")
    questions = load_questions(question_file, question_begin, question_end)
    if reference_file is not None:
        references = load_questions(reference_file, question_begin, question_end)

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
    print("start ans_handles append")
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
                if_reverse_answers,
                references[i : i + chunk_size] if reference_file is not None else None,
                if_fast_eval,
                answer_num
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
    if_reverse_answers,
    references,
    if_fast_eval,
    answer_num
):
    print("start load model")
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
        conv = conv_judge_pair.copy() if references is None else conv_judge_pair_w_reference.copy()
        template = conv.prompt_template

        # if fast eval, use the "\n" as the separator
        if if_fast_eval:
            conv.sep = "\n"

        # reverse the order of the answers
        if if_reverse_answers:
            temp_answer = question["answer1_body"]
            question["answer1_body"] = question["answer2_body"]
            question["answer2_body"] = temp_answer

        # combine data_sample
        if references is None:
            data_sample = conv.system + '\n' + template.format(question=question['question_body'],
                                                               answer_1=question['answer1_body'],
                                                               answer_2=question['answer2_body'],
                                                               prompt=conv.prompt) + conv.appendix
        else:
            data_sample = conv.system + '\n' + template.format(question=question['question_body'],
                                                               reference=references[q_i]['reference']['text'],
                                                               answer_1=question['answer1_body'],
                                                               answer_2=question['answer2_body'],
                                                               prompt=conv.prompt) + conv.appendix
        if answer_num > 2:
            plug_in_before_str = "[System]"
            plug_in_pos = data_sample.find(plug_in_before_str)

            new_answer = ""
            for i in range(2, int(answer_num)):
                new_answer += "[The Start of Assistant " + str(i+1) + "'s Answer]\n" + question[f'answer{i+1}_body'] + "\n\n" + "[The End of Assistant " + str(i+1) + "'s Answer]\n\n"

            new_data_sample = data_sample[:plug_in_pos] + new_answer + data_sample[plug_in_pos:]
            data_sample = new_data_sample

            data_sample = data_sample.replace(f"of two AI assistants", f"of {num2words[int(answer_num)]} AI assistants")
            data_sample = data_sample.replace("containing only two values indicating ", f"containing only {num2words[int(answer_num)]} values indicating ")
            data_sample = data_sample.replace("for Assistant 1 and 2", "for Assistant 1")

            plug_in_after_str = "for Assistant 1"
            plug_in_pos = data_sample.find(plug_in_after_str) + len(plug_in_after_str)

            new_answer = ""
            for i in range(int(answer_num)-2):
                new_answer += f", {i+2}"
            new_answer += f" and {int(answer_num)}"
            new_data_sample = data_sample[:plug_in_pos] + new_answer + data_sample[plug_in_pos:]
            data_sample = new_data_sample

            data_sample = data_sample.replace("The two scores are", f"The {num2words[int(answer_num)]} scores are")

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
            if references is not None:
                question["reference"] = references[q_i]['reference']['text']
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
        "--if-reverse-answers",
        type=int,
        default=0,
        help="Whether to reverse the order of the answers.",
    )
    parser.add_argument(
        "--reference-file",
        type=str,
        default=None,
        help="The reference file for evaluation.",
    )
    parser.add_argument(
        "--if-fast-eval",
        type=int,
        default=0,
        help="Whether to use fast evaluation.",
    )
    parser.add_argument(
        "--answer-num", 
        type=int,
        default=2,
        help="The number of answers."
    )
    args = parser.parse_args()
    args.if_reverse_answers = bool(args.if_reverse_answers)
    args.if_fast_eval = bool(args.if_fast_eval)
    if args.reference_file == 'None':
        args.reference_file = None
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
        args.if_reverse_answers,
        args.reference_file,
        args.if_fast_eval,
        args.answer_num
    )

    reorg_answer_file(args.answer_file)