#!/bin/bash
bash /client-tools/repair_A100.sh
cd /home/zhulianghui/ProjectC_ChatGPT/alpaca/reference/JudgeLM/;
source /opt/conda/bin/activate /home/zhulianghui/.conda/envs/jjlm-test

BASE_MODEL_PATH="/share/project/lianghuizhu/vicuna-weights-collection-v1.3/vicuna-7b-v1.3"
DATA_PATH="/home/zhulianghui/ProjectC_ChatGPT/alpaca/reference/JudgeLM/judgelm/data/JudgeLM/judgelm_train_100k.jsonl"

SWAP_AUG_RATIO=0.5
REF_DROP_RATIO=0.5

#######################################################################################################################

N_PROC_PER_NODE=4
NUM_TRAIN_EPOCHS=3
PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=32
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.
WARMUP_RATIO=0.03
LR_SCHEDULER_TYPE="cosine"
MODEL_MAX_LENGTH=2048

#RUN_NAME="vicuna-7b-v1.3-data-(review_output_test_0628_gpt4_all_30k_mix_instruct_all)-bs128-ep3-lr2e-5-wd0.0-wr0.03-cosine-mmlength2048"

MODEL_NAME=$(basename $BASE_MODEL_PATH)
DATASET_NAME=$(basename $DATA_PATH .jsonl)
TOTAL_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * N_PROC_PER_NODE * WORLD_SIZE))

RUN_NAME="${MODEL_NAME}-data(${DATASET_NAME})-bs${TOTAL_BATCH_SIZE}-ep${NUM_TRAIN_EPOCHS}-lr${LEARNING_RATE}-wd${WEIGHT_DECAY}-wr${WARMUP_RATIO}-${LR_SCHEDULER_TYPE}-mmlength${MODEL_MAX_LENGTH}"
RUN_NAME=${RUN_NAME//_/-}


if [[ "$SWAP_AUG_RATIO" != "-1.0" ]]; then
    RUN_NAME="${RUN_NAME}-swap_aug_ratio${SWAP_AUG_RATIO}"
fi

if [[ "$REF_DROP_RATIO" != "-1.0" ]]; then
    RUN_NAME="${RUN_NAME}-ref_drop_ratio${REF_DROP_RATIO}"
fi

RUN_NAME="${RUN_NAME}-debug++"

OUTPUT_PATH="/home/zhulianghui/ProjectC_ChatGPT/alpaca-quan/output/$RUN_NAME"
LOG_FILE="/home/zhulianghui/ProjectC_ChatGPT/alpaca-quan/logs/$RUN_NAME.log"


WANDB_MODE=offline torchrun --nproc_per_node=$N_PROC_PER_NODE --master_port=12313 --nnodes=${WORLD_SIZE} --node_rank=${RANK} --master_addr=${MASTER_ADDR}  judgelm/train/train_mem.py \
    --model_name_or_path=$BASE_MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit $NUM_TRAIN_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap offload" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --bf16 True \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --run_name $RUN_NAME \
    --swap_aug_ratio $SWAP_AUG_RATIO \
    --ref_drop_ratio $REF_DROP_RATIO
