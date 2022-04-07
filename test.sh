remark=dta
iter=100
skip_iter=-1
batch_size=2
max_src_len=128
max_tgt_len=128
learning_rate=2e-5
job_name=gen
export CUDA_VISIBLE_DEVICES=0

full_name=${job_name}.${remark}.max_src_len-${max_src_len}.max_tgt_len-${max_tgt_len}.batch_size-${batch_size}.learning_rate-${learning_rate}

model_dir=SavedModels/${full_name}/Epoch-0
vocab_dir=/data/pretrained_models/facebook/mbart-large-50
save_dir=SavedModels/${full_name}
log_file=${full_name}.test.log

echo "INFO: start testing model"

nohup python3 -u pipeline.py \
    --do_test \
    --src_lang zh_CN \
    --tgt_lang zh_CN \
    --max_src_len $max_src_len \
    --max_tgt_len $max_tgt_len \
    --test_filename data/dev.jsonl \
    --remark ${remark} \
    --pretrained_model_path ${model_dir} \
    --vocab_path ${vocab_dir} \
    --save_dir ${save_dir} \
    --batch_size ${batch_size} \
    --num_train_epochs ${iter} \
    --skip_eval_epochs ${skip_iter} \
    --learning_rate ${learning_rate} \
    >$log_file 2>&1 &
