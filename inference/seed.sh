# for dataset in msmarco-passage/trec-dl-2019 msmarco-passage/trec-dl-2020
window=20
step=$((window / 2))

echo "window: $window, step: $step"

for dataset in msmarco-passage/trec-dl-2019
do
    for seed in $( seq 0 19 )
    do
        # echo "seed: $seed"
        model=Qwen/Qwen2.5-7B-Instruct
        CUDA_VISIBLE_DEVICES=1 python sliding_window.py -model $model -d $dataset -window $window -step $step -seed $seed
        # CUDA_VISIBLE_DEVICES=1 python sliding_window.py -lora checkpoints/qwen2_5_7B/epoch_0 -d $dataset -window $window -step $step -seed $seed
        # CUDA_VISIBLE_DEVICES=1 python sliding_window.py -model Qwen/Qwen2.5-14B-Instruct -d $dataset -device 2
    done
done
