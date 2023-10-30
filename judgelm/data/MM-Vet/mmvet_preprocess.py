import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
root = file.parents[3]
sys.path.append(str(root))
print(sys.path)

import argparse
import os
import json

from judgelm.utils import extract_jsonl


def combine_mmvet_judge_samples(gt_file_path, pred_file_path):
    # load gt_file
    gt_dict = json.load(open(gt_file_path, 'r'))
    pred_dict = json.load(open(pred_file_path, 'r'))

    keys = list(gt_dict.keys())

    for i, key in enumerate(keys):
        gt_dict[key]["answer1_body"] = pred_dict[key]
        gt_dict[key]["question_id"] = i

    output_path = os.path.join(os.path.dirname(gt_file_path), "mm-vet-judge-samples.jsonl")

    # save new gt_dict as .jsonl
    with open(output_path, 'w') as f:
        for key in keys:
            f.write(json.dumps(gt_dict[key]) + '\n')

def translate_jsonl_to_md(answer_file, gt_dict=None):
    answer_list = extract_jsonl(answer_file)

    for answer in answer_list:
        # list items in gt_dict
        for item in gt_dict.items():
            print(item[1]["question_id"])
            if answer["question_id"] == item[1]["question_id"]:
                answer["answer"] = item[1]["answer"]
    
    with open(answer_file+'.md', "w") as fout:
        for answer in answer_list:
            print(answer)
            fout.write("### " + answer["imagename"] + " " + answer["question"] + '\n')
            fout.write("##### " + "Reference Answer" + '\n')
            fout.write(answer["answer"] + '\n')
            fout.write("##### " + "Answer1_body" + '\n')
            fout.write(answer["answer1_body"] + '\n')
            fout.write("##### " +"pred_text" + '\n')
            fout.write(answer["pred_text"] + '\n')
            fout.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_file_path', type=str, required=True, default="/home/zhulianghui/ProjectC_ChatGPT/alpaca/reference/JudgeLM/judgelm/data/MM-Vet/mm-vet-gt.json")
    parser.add_argument('--pred_file_path', type=str, required=True, default="/home/zhulianghui/ProjectC_ChatGPT/alpaca/reference/JudgeLM/judgelm/data/MM-Vet/mm-vet-emu-prediction.json")

    args = parser.parse_args()

    combine_mmvet_judge_samples(args.gt_file_path, args.pred_file_path)
