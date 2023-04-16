#!/bin/bash
#SBATCH -J run_selection_strategies         # name of job
#SBATCH -p cpu                              # name of partition or queue (default is cpu)
#SBATCH -o rss.out                          # name of output file for this submission script
#SBATCH -e rss.err                          # name of error file for this submission script

# sbatch --export=data_set_name='text_data_all' run_selection_strategies.sh

for h in random xpal alce qbc log-loss entropy
do
    for i in 7 9 124 343 456 536 925 293 864 783
    do
        echo $h $i
        python3 experimental_setup_csv.py \
            --query_strategy $h \
            --data_set $data_set_name \
            --results_path ../../results \
            --test_ratio 0.25 \
            --bandwidth mean \
            --budget 300 \
            --seed $i \
            --kernel cosine \
            &
    done
    wait
done

for j in 7 9 124 343 456 536 925 293 864 783
do
    echo "pal" $j
    python3 experimental_setup_csv.py \
        --query_strategy pal \
        --data_set $data_set_name \
        --results_path ../../results \
        --test_ratio 0.25 \
        --bandwidth mean \
        --budget 300 \
        --seed $j \
        --kernel cosine
done

# all_text_data_spilt
# for j in 7 12 23 34 45 56 67 78 89 100

# for h in random xpal qbc
# do
#     for i in 7 9 12 17 19 23 37 45 99 136  # old data split
#     do
#         echo $h $i
#         python3 experimental_setup_csv.py \
#             --query_strategy $h \
#             --data_set all_data_filtered \
#             --results_path ../../results \
#             --test_ratio 0.25 \
#             --bandwidth mean \
#             --budget 415 \
#             --seed $i \
#             --kernel cosine \
#             &
#     done
#     wait
# done