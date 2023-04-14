# rbf kernel

python experimental_setup_csv.py \
  --query_strategy xpal \
  --data_set tfidf_data \
  --results_path ../../results \
  --test_ratio 0.25 \
  --bandwidth mean \
  --budget 612 \
  --seed 1 \
  --kernel rbf

python experimental_setup_csv.py \
  --query_strategy  qbc \
  --data_set tfidf_data \
  --results_path ../../results \
  --test_ratio 0.25 \
  --bandwidth mean \
  --budget 612 \
  --seed 1 \
  --kernel rbf

python experimental_setup_csv.py \
  --query_strategy  random \
  --data_set tfidf_data \
  --results_path ../../results \
  --test_ratio 0.25 \
  --bandwidth mean \
  --budget 612 \
  --seed 1 \
  --kernel rbf

python experimental_setup_csv.py \
  --query_strategy pal \
  --data_set tfidf_data \
  --results_path ../../results \
  --test_ratio 0.25 \
  --bandwidth mean \
  --budget 612 \
  --seed 1 \
  --kernel rbf
  
  # cosine kernel
  
python experimental_setup_csv.py \
  --query_strategy xpal \
  --data_set tfidf_data \
  --results_path ../../results \
  --test_ratio 0.25 \
  --bandwidth mean \
  --budget 612 \
  --seed 1 \
  --kernel cosine

python experimental_setup_csv.py \
  --query_strategy  qbc \
  --data_set tfidf_data \
  --results_path ../../results \
  --test_ratio 0.25 \
  --bandwidth mean \
  --budget 612 \
  --seed 1 \
  --kernel cosine

python experimental_setup_csv.py \
  --query_strategy  random \
  --data_set tfidf_data \
  --results_path ../../results \
  --test_ratio 0.25 \
  --bandwidth mean \
  --budget 612 \
  --seed 1 \
  --kernel cosine

python experimental_setup_csv.py \
  --query_strategy pal \
  --data_set tfidf_data \
  --results_path ../../results \
  --test_ratio 0.25 \
  --bandwidth mean \
  --budget 612 \
  --seed 1 \
  --kernel cosine

python experimental_setup_csv.py \
  --query_strategy random \
  --data_set text_data_all \
  --results_path ../../results \
  --test_ratio 0.25 \
  --bandwidth mean \
  --budget 200 \
  --seed 7 \
  --kernel cosine 