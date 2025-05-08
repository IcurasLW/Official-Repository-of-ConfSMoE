
CUDA_VISIBLE_DEVICES=3 python ../main.py \
    --data mimic4 \
    --modality INTE \
    --batch_size 256 \
    --num_experts 4 \
    --top_k 2 \
    --label 25 \
    --seq_len 48 \
    --dropout 0.2 \
    --task phenotyping \
    --lr 3e-4 \
    --multilabel True \
    --epochs 50 \
    --TokenLevelConf False 