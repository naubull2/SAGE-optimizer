#!/bin/bash
set -e

# --- 1. Define Experiment Matrix ---
BASE_OUTPUT_DIR="./outputs/SAGE_llama_exp"
TOKENIZER_PATH="./model/llama/"
DATASET_PATH="./data/dsir-pile" # Update if this path changed
MAX_STEPS=50000

MODELS=(
    "llama_60m"
    "llama_130m"
    "llama_270m"
    "llama_0.6b"
    "llama_1.3b"
)

OPTIMIZERS=(
    "AdamW"
    "Lion"
    "APOLLO"
    "SinkGD_pure"
    "SinkGD"
    "SAGE_pure"        # pure SAGE in O(N)
    "SAGE_lion"
    "SAGE_hybrid"
)

SEEDS=(42 88 123)

BATCH_SIZE=64
GRAD_ACC_STEP=4

# --- 2. Run Experiment Loops ---
echo "Starting Experiment Matrix: ${#MODELS[@]} models x ${#OPTIMIZERS[@]} optimizers x ${#SEEDS[@]} seeds"
echo "---"

for model_size in "${MODELS[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
        
        # --- Set Optimizer-Specific Hyperparameters ---
        case $optimizer in
            "AdamW")
                LR="1e-4"
                BETAS="0.9,0.999"
                WD="0.01"
                SINKHORN_SCALE=1
                ;;
            "Lion" | "SAGE_lion")
                LR="1e-4"
                BETAS="0.9,0.99"
                WD="0.01"
                SINKHORN_SCALE=100
                ;;
            "APOLLO")
                LR="1e-4"
                BETAS="0.9,0.99"
                WD="0.01"
                SINKHORN_SCALE=1
                ;;
            "SinkGD_pure" | "SinkGD")
                LR="2e-4"
                BETAS="0.9,0.99"
                WD="0.01"
                SINKHORN_SCALE=100
                ;;
            "SAGE_pure" | "SAGE_hybrid")
                LR="2e-3"
                BETAS="0.9,0.99"
                WD="0.01"
                SINKHORN_SCALE=10
                ;;
        esac

        for seed in "${SEEDS[@]}"; do
            
            RUN_NAME="${model_size}_${optimizer}"
            MODEL_CONFIG_PATH="./configs/${model_size}"
            OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}/seed_${seed}"
            
            echo "--- STARTING RUN ---"
            echo "MODEL: $model_size"
            echo "OPTIMIZER: $optimizer"
            echo "SEED: $seed"
            echo "LR: $LR, Betas: $BETAS, WD: $WD"
            echo "OUTPUT_DIR: $OUTPUT_DIR"
            echo "---"

            python train.py \
                --model_name_or_path "$MODEL_CONFIG_PATH" \
                --tokenizer_name "$TOKENIZER_PATH" \
                --dataset_path "$DATASET_PATH" \
                --optimizer "$optimizer" \
                \
                --output_dir "$OUTPUT_DIR" \
                --logging_dir "${OUTPUT_DIR}/logs" \
                --run_name "${RUN_NAME}_seed${seed}" \
                \
                --max_steps $MAX_STEPS \
                --learning_rate $LR \
                --weight_decay $WD \
                --betas $BETAS \
                --sinkhorn_scale $SINKHORN_SCALE \
                --seed $seed \
                \
                --per_device_train_batch_size $BATCH_SIZE \
                --per_device_eval_batch_size $BATCH_SIZE \
                --gradient_accumulation_steps $GRAD_ACC_STEP \
                --eval_strategy steps \
                --eval_steps 1000 \
                --logging_steps 10 \
                --save_strategy no \
                --warmup_ratio 0.1 \
                --lr_scheduler_type cosine \
                --eval_on_start true \
                --is_pretrain true \
                --report_to none

            echo "--- Finished: ${RUN_NAME}_seed${seed} ---"
            echo ""
            
        done
    done
done

echo "All experiments complete."
