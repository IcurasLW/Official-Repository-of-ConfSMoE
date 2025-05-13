

for ratio in 0 0.1 0.2 0.3 0.4 0.5
do
    CUDA_VISIBLE_DEVICES=1 python ../main.py \
        --modality 'AIN' \
        --data CMU_MOSI \
        --missing_ratio "$ratio" \
        --batch_size 256 \
        --num_experts 8 \
        --top_k 2 \
        --label 3 \
        --seq_len 50 \
        --dropout 0.1 \
        --datapath data/CMU-MOSI/Processed/ \
        --TokenLevelConf True
done

for ratio in 0 0.1 0.2 0.3 0.4 0.5
do
    CUDA_VISIBLE_DEVICES=1 python ../main.py \
        --modality 'AIN' \
        --data CMU_MOSI \
        --missing_ratio "$ratio" \
        --batch_size 256 \
        --num_experts 8 \
        --top_k 2 \
        --label 3 \
        --seq_len 50 \
        --dropout 0.1 \
        --datapath data/CMU-MOSI/Processed/ \
        --TokenLevelConf False
done

