lang=ruby
python3 run.py \
    --output_dir saved_models/$lang \
    --model_name_or_path microsoft/unixcoder-base  \
    --lang=$lang \
    --do_test \
    --eval_data_file ./dataset/$lang/valid.jsonl \
    --test_data_file ./dataset/$lang/test.jsonl \
    --codebase_file ./dataset/$lang/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --ast_length 320  \
    --dfg_length 128  \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --seed 123456