import argparse


def add_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_src_len", default=128, type=int, help="")
    parser.add_argument("--max_tgt_len", default=16, type=int, help="")
    parser.add_argument("--batch_size", default=32, type=int, help="")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="")
    parser.add_argument("--skip_eval_epochs", default=-1, type=int, help="")
    parser.add_argument("--seed", default=42, type=int, help="")

    parser.add_argument("--pretrained_model_path",
                        default=None, type=str, help="")
    parser.add_argument("--vocab_path",
                        default=None, type=str, help="")
    parser.add_argument("--train_filename",
                        default="data/train.jsonl", type=str, help="")
    parser.add_argument("--val_filename",
                        default="data/dev.jsonl", type=str, help="")
    parser.add_argument("--test_filename",
                        default="data/test.jsonl", type=str, help="")
    parser.add_argument("--save_dir", default=None,
                        required=True, type=str, help="")
    parser.add_argument("--remark", default="", type=str)
    parser.add_argument("--src_lang", default="zh_CN", type=str)
    parser.add_argument("--tgt_lang", default="zh_CN", type=str)

    parser.add_argument("--do_train", action="store_true", help="")
    parser.add_argument("--do_eval", action="store_true", help="")
    parser.add_argument("--do_test", action="store_true", help="")
    args = parser.parse_args()

    return args
