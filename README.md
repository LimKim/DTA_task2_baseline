# DTA_task2_baseline
Baseline method for DTA Dialogue Summarization task.

## Setup

### Install dependencies
Please install all the dependency packages using the following command:
```
pip install -r requirements.txt
```

## Quick Start
1. Download [dataset](https://pan.baidu.com/s/1nEZfBJIwFpGqidDKfWromQ?errmsg=Auth+Login+Params+Not+Corret&errno=2&ssnerror=0#list/path=%2F) (passord: 7mvn) and unzip it under the `data` folder.
2. Download the pretrained model [mbart-large-50](https://huggingface.co/facebook/mbart-large-50).
3. Execute command `python3 preprocess.py` to generate data for model training. This will generate `train.jsonl` and `dev.jsonl` in the `data` folder. Note that it will take few minutes.
4. You can execute `bash train.sh` or the following command to train an baseline model.
    ```bash
    python3 -u pipeline.py \
        --do_train \
        --do_eval \
        --src_lang zh_CN \
        --tgt_lang zh_CN \
        --train_filename data/train.jsonl \
        --val_filename data/dev.jsonl \
        --max_src_len ${max_src_len} \
        --max_tgt_len ${max_tgt_len} \
        --remark ${remark} \
        --pretrained_model_path ${save_dir} \
        --vocab_path ${vocab_dir} \
        --save_dir ${save_dir} \
        --batch_size ${batch_size} \
        --num_train_epochs ${iter} \
        --skip_eval_epochs ${skip_iter} \
        --learning_rate ${learning_rate}
    ```
5. You can execute `bash test.sh` or the following command to generate dialogue summary by the model trained before.  And you will get your generated results in the `test.pred` file.
    ```bash
    python3 -u pipeline.py \
        --do_test \
        --src_lang zh_CN \
        --tgt_lang zh_CN \
        --test_filename data/dev.jsonl \
        --max_src_len ${max_src_len} \
        --max_tgt_len ${max_tgt_len} \
        --remark ${remark} \
        --pretrained_model_path ${model_dir} \
        --vocab_path ${vocab_dir} \
        --save_dir ${save_dir} \
        --batch_size ${batch_size} \
        --num_train_epochs ${iter} \
        --skip_eval_epochs ${skip_iter} \
        --learning_rate ${learning_rate}
    ```


## Bug or Questions?
If you have any questions about the code, please open an issue or contact us by <yifan.yang@aispeech.com>. 

Please try to specify the problem with details so we can help you better and quicker!