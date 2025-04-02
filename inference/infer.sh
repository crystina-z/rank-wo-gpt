# for dataset in msmarco-passage/trec-dl-2019 msmarco-passage/trec-dl-2020
window=40
step=$((window / 2))

echo "window: $window, step: $step"

for dataset in msmarco-passage/trec-dl-2019
do
    CUDA_VISIBLE_DEVICES=1 python sliding_window.py -lora checkpoints/qwen2_5_7B/epoch_0 -d $dataset -window $window -step $step
    # CUDA_VISIBLE_DEVICES=1 python sliding_window.py -model Qwen/Qwen2.5-14B-Instruct -d $dataset -device 2
done
