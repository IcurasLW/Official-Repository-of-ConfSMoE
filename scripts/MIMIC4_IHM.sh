
CUDA_VISIBLE_DEVICES=1 python  ../main.py \
    --data mimic4 \
    --modality INTE \
    --batch_size 256 \
    --num_experts 4 \
    --top_k 2 \
    --label 2 \
    --seq_len 48 \
    --dropout 0.1 \
    --task in-hospital-mortality \
    --lr 3e-4 \
    --multilabel False \
    --epochs 50 \
    --TokenLevelConf False \