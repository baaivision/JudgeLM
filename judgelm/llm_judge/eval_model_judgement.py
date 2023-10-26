import argparse
import json
import os
import time

import shortuuid
import torch
from tqdm import tqdm

import sys
from pathlib import Path  # if you haven't already done so

file = Path(__file__).resolve()
root = file.parents[2]
sys.path.append(str(root))
print(sys.path)

from judgelm.utils import extract_jsonl


from judgelm.llm_judge.common import parse_score, translate_score_to_win_list


def filt_pred_score_list_by_gt_score_list(gt_score_list, pred_score_list):
    new_gt_score_list = []
    new_pred_score_list = []
    # filter [-1, -1] pairs
    for gt_score, pred_score in zip(gt_score_list, pred_score_list):
        if gt_score[0] == -1 or gt_score[1] == -1:
            continue
        else:
            new_gt_score_list.append(gt_score)
            new_pred_score_list.append(pred_score)

    return new_gt_score_list, new_pred_score_list


def calculate_metrics(gt_answer_file_path, sequential_pred_answer_file_path, reversed_pred_answer_file_path, if_filter_minus_one=True):
    # get file list
    gt_answer_file_list = extract_jsonl(gt_answer_file_path)  # [:1000]
    # check if sequential_pred_answer_file_path is `str`
    if isinstance(sequential_pred_answer_file_path, str):
        sequential_pred_answer_file_list = extract_jsonl(sequential_pred_answer_file_path)  # [:1000]
    elif (isinstance(sequential_pred_answer_file_path, list)):
        sequential_pred_answer_file_list = sequential_pred_answer_file_path
    else:
        pass

    gt_score_list = []
    for gt_answer_file in gt_answer_file_list:
        gt_score_list.append(parse_score(gt_answer_file['text']))
    print(
        "===============================================================================================================")
    sequential_pred_score_list = []
    for sequential_pred_answer_file in sequential_pred_answer_file_list:
        sequential_pred_score_list.append(parse_score(sequential_pred_answer_file['pred_text']))

    gt_score_list_contains_minus_one = gt_score_list.copy()

    if if_filter_minus_one:
        # filter pred by gt
        gt_score_list, sequential_pred_score_list = filt_pred_score_list_by_gt_score_list(gt_score_list,
                                                                                      sequential_pred_score_list)

    # win_list calculate v2
    # if the score gap is less than T, we consider it as a draw
    T = 0.0
    gt_win_list = translate_score_to_win_list(gt_score_list, T)
    sequential_pred_win_list = translate_score_to_win_list(sequential_pred_score_list, T)

    # sklearn.metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_true = gt_win_list
    y_pred = sequential_pred_win_list

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred, average='macro')

    recall = recall_score(y_true, y_pred, average='macro')

    f1 = f1_score(y_true, y_pred, average='macro')

    # add metrics to dict
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    if isinstance(reversed_pred_answer_file_path, str):
        reversed_pred_answer_file_list = extract_jsonl(reversed_pred_answer_file_path)
    elif (isinstance(reversed_pred_answer_file_path, list)):
        reversed_pred_answer_file_list = reversed_pred_answer_file_path
    else:
        pass

    reversed_pred_score_list = []
    for reversed_pred_answer_file in reversed_pred_answer_file_list:
        reversed_pred_score_list.append(parse_score(reversed_pred_answer_file['pred_text']))
    _, reversed_pred_score_list = filt_pred_score_list_by_gt_score_list(gt_score_list_contains_minus_one,
                                                                        reversed_pred_score_list)
    reversed_pred_win_list = translate_score_to_win_list(reversed_pred_score_list, T)

    # from the perspect of before one
    # 1: win, -1: lose, 0: draw

    same = 0
    perfer_before = 0
    perfer_after = 0
    for i in range(len(sequential_pred_win_list)):
        if sequential_pred_win_list[i] == 1:
            sequential_pred_win_list[i] = -1
        elif sequential_pred_win_list[i] == -1:
            sequential_pred_win_list[i] = 1
        else:
            pass
        if sequential_pred_win_list[i] == reversed_pred_win_list[i]:
            same += 1
        elif sequential_pred_win_list[i] - reversed_pred_win_list[i] > 0:  # 1 & 0， 1 & -1， 0 & -1
            perfer_after += 1
            # print(i)
        elif sequential_pred_win_list[i] - reversed_pred_win_list[i] < 0:  # -1 & 0， -1 & 1， 0 & 1
            perfer_before += 1
            # print(i)
        else:
            pass

    # add metrics to dict
    metrics_dict['consistency'] = same / len(sequential_pred_win_list)
    metrics_dict['perfer_before_rate'] = perfer_before / len(sequential_pred_win_list)
    metrics_dict['perfer_after_rate'] = perfer_after / len(sequential_pred_win_list)
    metrics_dict['delta_bias'] = abs(perfer_before - perfer_after) / len(sequential_pred_win_list)
    metrics_dict['total_bias'] = (perfer_before + perfer_after) / len(sequential_pred_win_list)
    metrics_dict['total_num'] = len(sequential_pred_win_list)

    return metrics_dict


if __name__ == '__main__':
    # gt w/o ref
    gt_answer_file_path = "/share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgelm/data/JudgeLM/judgelm_val_5k_gpt4.jsonl"
    # gt_answer_file_path = "/home/zhulianghui/ProjectC_ChatGPT/alpaca-quan/eval/table/review_output_test_0627_gpt4_1st_5000_val_mix_instruct.jsonl"
    # gt w/ ref
    # gt_answer_file_path = "/share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgelm/data/JudgeLM/judgelm_val_5k_gpt4_with_reference.jsonl"

    # 33b 100k full model lr 3e-5
    sequential_pred_answer_file_path = "/home/zhulianghui/ProjectC_ChatGPT/alpaca/reference/FastChat/vicuna-33b-v1.3-data(judgelm-train-0628-gpt4-100k-w-reference-all-w-reference-drop)-bs128-ep3-lr3e-5-wd0.-wr0.03-cosine-mmlength2048-lazy-preprocess-swap-aug-ref-drop-ratio0.5-val-data-prepared-sampled-0627"
    sequential_pred_answer_file_path = "/home/zhulianghui/ProjectC_ChatGPT/alpaca/reference/FastChat/vicuna-7b-v1.3-data(judgelm-train-0628-gpt4-100k-w-reference-all-w-reference-drop)-bs128-ep3-lr2e-5-wd0.-wr0.03-cosine-mmlength2048-lazy-preprocess-swap-aug-ref-drop-ratio0.5-val-data-prepared-sampled-0627"
    sequential_pred_answer_file_path = "/home/zhulianghui/ProjectC_ChatGPT/alpaca/reference/FastChat/vicuna-33b-v1.3-data(judgelm-train-0628-gpt4-100k-w-reference-all-w-reference-drop)-bs128-ep3-lr1e-5-wd0.-wr0.03-cosine-mmlength2048-lazy-preprocess-swap-aug-ref-drop-ratio0.5-val-data-prepared-sampled-0627"
    sequential_pred_answer_file_path = "/home/zhulianghui/ProjectC_ChatGPT/alpaca/reference/FastChat/vicuna-33b-v1.3-data(judgelm-train-0628-gpt4-100k-w-reference-all-w-reference-drop)-bs128-ep3-lr1e-5-wd0.-wr0.03-cosine-mmlength2048-lazy-preprocess-swap-aug-ref-drop-ratio0.5-val-data-prepared-sampled-0627-reference"
    sequential_pred_answer_file_path = "/share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgements_output/JudgeLM/7b-full-model-pycharm-debug-v2"
    #
    # 33b 100k full model lr 3e-5
    reversed_pred_answer_file_path = "/home/zhulianghui/ProjectC_ChatGPT/alpaca/reference/FastChat/vicuna-33b-v1.3-data(judgelm-train-0628-gpt4-100k-w-reference-all-w-reference-drop)-bs128-ep3-lr3e-5-wd0.-wr0.03-cosine-mmlength2048-lazy-preprocess-swap-aug-ref-drop-ratio0.5-val-data-prepared-sampled-0627-reverse"
    reversed_pred_answer_file_path = "/home/zhulianghui/ProjectC_ChatGPT/alpaca/reference/FastChat/vicuna-7b-v1.3-data(judgelm-train-0628-gpt4-100k-w-reference-all-w-reference-drop)-bs128-ep3-lr2e-5-wd0.-wr0.03-cosine-mmlength2048-lazy-preprocess-swap-aug-ref-drop-ratio0.5-val-data-prepared-sampled-0627-reverse"
    reversed_pred_answer_file_path = "/home/zhulianghui/ProjectC_ChatGPT/alpaca/reference/FastChat/vicuna-33b-v1.3-data(judgelm-train-0628-gpt4-100k-w-reference-all-w-reference-drop)-bs128-ep3-lr1e-5-wd0.-wr0.03-cosine-mmlength2048-lazy-preprocess-swap-aug-ref-drop-ratio0.5-val-data-prepared-sampled-0627-reverse"
    reversed_pred_answer_file_path = "/home/zhulianghui/ProjectC_ChatGPT/alpaca/reference/FastChat/vicuna-33b-v1.3-data(judgelm-train-0628-gpt4-100k-w-reference-all-w-reference-drop)-bs128-ep3-lr1e-5-wd0.-wr0.03-cosine-mmlength2048-lazy-preprocess-swap-aug-ref-drop-ratio0.5-val-data-prepared-sampled-0627-reverse-reference"
    reversed_pred_answer_file_path = "/home/zhulianghui/ProjectC_ChatGPT/alpaca/reference/JudgeLM/judgements_output/JudgeLM/7b-full-model-pycharm-debug-reverse-v2"

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-answer-file-path", type=str, default=gt_answer_file_path)
    parser.add_argument("--sequential-pred-answer-file-path", type=str, default=sequential_pred_answer_file_path)
    parser.add_argument("--reversed-pred-answer-file-path", type=str, default=reversed_pred_answer_file_path)

    args = parser.parse_args()

    metrics_dict = calculate_metrics(args.gt_answer_file_path, args.sequential_pred_answer_file_path,
                                     args.reversed_pred_answer_file_path)

    print(metrics_dict)
