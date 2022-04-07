import json
from rouge import Rouge
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.sampler import SequentialSampler

default_src_column = "src_txt"
default_tgt_column = "tgt_txt"


class CustomDataset(Dataset):
    def __init__(self, filename, readfile_method, write2file=False):
        self.x = readfile_method(filename=filename, write2file=write2file)
        self.length = len(self.x)

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.length


class MBartForConditionalGenerationProcessor():
    def read_file(self, filename, write2file):
        fr = open(filename, "r", encoding="utf8")
        merge_data = []
        orig_data = [json.loads(line) for line in fr] if filename.endswith(
            "jsonl") else json.load(fr)
        for jline in orig_data:
            src_text = jline[default_src_column]
            tgt_text = jline[default_tgt_column]
            data = {"src_text": src_text, "tgt_text": tgt_text}
            merge_data.append(data)
        return merge_data

    def convert_data_for_bart(self, data, tokenizer, args, src_lang=None, tgt_lang=None):
        src_list = [one_data["src_text"] for one_data in data]
        tgt_list = [one_data["tgt_text"] for one_data in data]

        model_inputs = tokenizer(
            src_list,
            return_tensors="pt",
            max_length=args.max_src_len,
            truncation=True,
            padding=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                tgt_list,
                return_tensors="pt",
                max_length=args.max_tgt_len,
                truncation=True,
                padding=True).input_ids

        return model_inputs, labels

    def get_dataloader(self, dataset, args, tokenizer, random=True):
        def collate_fn(data):
            return self.convert_data_for_bart(data, tokenizer=tokenizer, args=args)

        if random is False:
            _sampler = SequentialSampler(dataset)
        else:
            _sampler = RandomSampler(dataset)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            sampler=_sampler,
            collate_fn=collate_fn
        )
        return dataloader

    def metrics_report(self, reference, hypothesis):
        if reference == []:
            report = "Reference is empty, skip do metrics"
            focus_score = 0
            return report, focus_score

        rouge = Rouge()
        _hypothesis = [" ".join(list(sent)) for sent in hypothesis]
        _reference = [" ".join(list(sent)) for sent in reference]

        report = rouge.get_scores(_hypothesis, _reference, avg=True)
        focus_score = (report["rouge-1"]["f"] +
                       report["rouge-2"]["f"] +
                       report["rouge-l"]["f"])/3
        report = json.dumps(report, ensure_ascii=False, indent=4)
        return report, focus_score
