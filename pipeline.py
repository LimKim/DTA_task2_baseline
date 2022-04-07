import json
import logging
import os
import random

import numpy as np
import torch
from transformers import (AdamW, MBart50TokenizerFast,
                          MBartForConditionalGeneration)
from args import add_argument
from ModelUtils import CustomDataset, MBartForConditionalGenerationProcessor

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def pipeline():
    args = add_argument()
    logger.info(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = MBartForConditionalGeneration.from_pretrained(
        args.pretrained_model_path)
    tokenizer = MBart50TokenizerFast.from_pretrained(
        args.vocab_path,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang)
    processor = MBartForConditionalGenerationProcessor()
    model.cuda()

    optimizer_grouped_parameters = model.parameters()
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    if args.do_train:
        train_path = os.path.join(args.train_filename)
        dev_path = os.path.join(args.val_filename)
        train_dataset = CustomDataset(train_path, processor.read_file)
        dev_dataset = CustomDataset(dev_path, processor.read_file)
        train_op(
            model,
            processor,
            train_dataset,
            dev_dataset,
            args,
            tokenizer,
            optimizer
        )
    if args.do_test:
        test_path = os.path.join(args.test_filename)
        test_dataset = CustomDataset(
            test_path, processor.read_file, write2file=True)
        eval_op(model,
                processor, test_dataset, args, tokenizer, write2file=True, remark=args.remark)


def train_op(model, processor, train_dataset, dev_dataset, args, tokenizer, optimizer):
    model.train()
    train_dataloader = processor.get_dataloader(
        dataset=train_dataset,
        args=args,
        tokenizer=tokenizer
    )

    num_steps = len(train_dataloader)
    max_score = 0
    print_per_step = max(10, int(num_steps / 10))

    for epoch in range(args.num_train_epochs):
        max_loss, min_loss = 0, 100
        for step, batch_data in enumerate(train_dataloader):
            model_inputs, labels = batch_data
            model_inputs["input_ids"] = model_inputs["input_ids"].cuda()
            model_inputs["attention_mask"] = model_inputs["attention_mask"].cuda()
            labels = labels.cuda()

            loss = model(**model_inputs, labels=labels).loss
            loss.backward()

            if step % print_per_step == 0:
                logger.info(
                    "Train:: EPOCH: {0}, Step: {1}/{2}, loss: {3}".format(
                        epoch, step, num_steps, round(loss.item(), 8)
                    )
                )
            if loss.item() < min_loss:
                min_loss = loss.item()
            if loss.item() > max_loss:
                max_loss = loss.item()

            optimizer.step()
            optimizer.zero_grad()
        logger.info("Epoch {} loss in range ({}, {})".format(
            epoch, round(min_loss, 4), round(max_loss, 4)))
        # eval the model
        if args.do_eval and epoch > args.skip_eval_epochs:
            focus_score = eval_op(
                model, processor, dev_dataset, args, tokenizer)

            logger.info("Best score: {}, Current score: {}".format(
                max_score, focus_score))

            if focus_score > max_score:
                max_score = focus_score
                # save the model
                logger.info(
                    "A New record exist, Save to dir: EPOCH-{0}".format(epoch))
                epoch_model_save_dir = os.path.join(
                    args.save_dir, "Epoch-{0}".format(epoch))
                model.save_pretrained(epoch_model_save_dir)


def eval_op(model, processor, test_dataset, args, tokenizer, write2file=False, remark=""):
    model.eval()
    test_dataloader = processor.get_dataloader(
        dataset=test_dataset,
        args=args,
        random=False,
        tokenizer=tokenizer,
    )
    num_steps = len(test_dataloader)
    print_per_step = max(10, int(num_steps / 10))
    hypothesis = []
    reference = [sent["tgt_text"] for sent in test_dataset.x]
    for step, batch_data in enumerate(test_dataloader):
        model_inputs, labels = batch_data
        model_inputs["input_ids"] = model_inputs["input_ids"].cuda()

        if step % print_per_step == 0:
            logger.info("Evaluate:: Step: {0}/{1}".format(step, num_steps))

        with torch.no_grad():
            summary_ids = model.generate(
                model_inputs["input_ids"], num_beams=4, max_length=args.max_tgt_len)
            preds = tokenizer.batch_decode(
                summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        hypothesis.extend(preds)

    if write2file:
        fw = open("test.pred", "w")
        json.dump(hypothesis, fw, ensure_ascii=False)

    report, focus_score = processor.metrics_report(
        reference, hypothesis)
    logger.info("\n%s" % report)
    return focus_score


if __name__ == "__main__":
    pipeline()
