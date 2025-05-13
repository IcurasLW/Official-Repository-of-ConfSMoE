

for ratio in 0 0.1 0.2 0.3 0.4 0.5
do
    CUDA_VISIBLE_DEVICES=2 python ../main.py \
        --modality AIN \
        --data CMU_MOSEI \
        --missing_ratio "$ratio" \
        --batch_size 256 \
        --num_experts 8 \
        --top_k 2 \
        --label 3 \
        --seq_len 50 \
        --dropout 0.1 \
        --datapath 'data/CMU-MOSEI/Processed/' \
        --TokenLevelConf True
done


for ratio in 0 0.1 0.2 0.3 0.4 0.5
do
    CUDA_VISIBLE_DEVICES=2 python ../main.py \
        --modality AIN \
        --data CMU_MOSEI \
        --missing_ratio "$ratio" \
        --batch_size 256 \
        --num_experts 8 \
        --top_k 2 \
        --label 3 \
        --seq_len 50 \
        --dropout 0.1 \
        --datapath 'data/CMU-MOSEI/Processed/' \
        --TokenLevelConf False
done

# for mod in AN AI IN A N I
# do
#     CUDA_VISIBLE_DEVICES=0 python ../main.py \
#         --data CMU_MOSEI \
#         --missing_ratio 0.5 \
#         --modality "$mod" \
#         --batch_size 256 \
#         --num_experts 16 \
#         --top_k 2 \
#         --label 3 \
#         --seq_len 50 \
#         --dropout 0.1 \
#         --datapath 'data/CMU-MOSEI/Processed/' \
#         --TokenLevelConf False
# done


# for mod in AN AI IN A N I
# do
#     CUDA_VISIBLE_DEVICES=0 python ../main.py \
#         --data CMU_MOSEI \
#         --missing_ratio 0.5 \
#         --modality "$mod" \
#         --batch_size 256 \
#         --num_experts 16 \
#         --top_k 2 \
#         --label 3 \
#         --seq_len 50 \
#         --dropout 0.1 \
#         --datapath 'data/CMU-MOSEI/Processed/' \
#         --TokenLevelConf True
# done

