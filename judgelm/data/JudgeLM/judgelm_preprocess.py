import sys
from pathlib import Path  # if you haven't already done so

file = Path(__file__).resolve()
root = file.parents[3]
sys.path.append(str(root))
print(sys.path)

import argparse
import os
import json

from judgelm.utils import extract_jsonl, save_jsonl


def combine_judgelm_val_judge_samples(ans1_file_path, ans2_file_path):
    # load gt_file
    ans1_dict_list = extract_jsonl(ans1_file_path)
    ans2_dict_list = extract_jsonl(ans2_file_path)

    sample_list = []
    for (ans1_dict, ans2_dict) in zip(ans1_dict_list, ans2_dict_list):
        assert ans1_dict['question_id'] == ans2_dict['question_id']
        i = ans1_dict['question_id']
        assert ans1_dict['question_body'] == ans2_dict['question_body']
        question_body = ans1_dict['question_body']

        sample_dict = {
            'question_id': i,
            'score': [ans1_dict['scores'], ans2_dict['scores']],
            'question_body': question_body,
            'answer1_body': ans1_dict['text'],
            'answer2_body': ans2_dict['text'],
            'answer1_model_id': ans1_dict['model'],
            'answer2_model_id': ans2_dict['model'],
            'answer1_metadata': {
                'decoding_method': ans1_dict['decoding_method'],
            },
            'answer2_metadata': {
                'decoding_method': ans2_dict['decoding_method'],
            }
        }
        sample_list.append(sample_dict)

    output_path = os.path.join(os.path.dirname(os.path.dirname(ans1_file_path)), "judgelm-val-5k-judge-samples.jsonl")

    save_jsonl(sample_list, output_path)


def translate_jsonl_to_md(answer_file, gt_dict=None):
    answer_list = extract_jsonl(answer_file)

    for answer in answer_list:
        # list items in gt_dict
        for item in gt_dict.items():
            print(item[1]["question_id"])
            if answer["question_id"] == item[1]["question_id"]:
                answer["answer"] = item[1]["answer"]

    with open(answer_file + '.md', "w") as fout:
        for answer in answer_list:
            print(answer)
            fout.write("### " + answer["imagename"] + " " + answer["question"] + '\n')
            fout.write("##### " + "Reference Answer" + '\n')
            fout.write(answer["answer"] + '\n')
            fout.write("##### " + "Answer1_body" + '\n')
            fout.write(answer["answer1_body"] + '\n')
            fout.write("##### " + "pred_text" + '\n')
            fout.write(answer["pred_text"] + '\n')
            fout.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ans1_file_path', type=str, required=True,
                        default="/home/zhulianghui/ProjectC_ChatGPT/alpaca/reference/JudgeLM/judgelm/data/JudgeLM/answers/alpaca_judgelm_val.jsonl")
    parser.add_argument('--ans2_file_path', type=str, required=True,
                        default="/home/zhulianghui/ProjectC_ChatGPT/alpaca/reference/JudgeLM/judgelm/data/JudgeLM/answers/vicuna_judgelm_val.jsonl")

    args = parser.parse_args()

    combine_judgelm_val_judge_samples(args.ans1_file_path, args.ans2_file_path)
