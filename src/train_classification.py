import argparse
import inspect
import json
import logging
import math
import os
import random
import evaluate
import transformers
import torch
import torch.optim as optim
import yaml
from tqdm import tqdm

from accelerate import load_checkpoint_and_dispatch, init_empty_weights, Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datetime import datetime
from collections import defaultdict
from transformers import default_data_collator
from transformers import get_scheduler
from torch.utils.data import DataLoader

from src.magnet import MagnetTransformerLM, MagnetAverageSingleInputWithPadding
from src.utils import read_json_file, init_seed, save_args_to_json, calculate_mean
from src.data_utils import MixtureByteVocab, JointInputcorpus


logger = get_logger(__name__)

def load_pretrained_model(model_path, model_type):
    config_file = os.path.join(model_path, "config.json")
    model_ckpt = os.path.join(model_path, "model.pth")
    config = read_json_file(config_file)

    # Load Pretrained model
    def get_model_config():
        model_args = inspect.getfullargspec(MagnetTransformerLM).args
        assert model_args.index('self') == 0
        model_args = model_args[1:]

        return {arg: config.get(arg) for arg in model_args}

    pretrained_model = MagnetTransformerLM(**get_model_config())
    pretrained_model.load_state_dict(torch.load(model_ckpt)["model"])

    return pretrained_model,  config

def parse_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parser = argparse.ArgumentParser(parents=[parent_parser])
    cfg_parser = argparse.ArgumentParser(parents=[parent_parser])

    cfg_parser.add_argument('--config', default='default')
    cfg_parser.add_argument('--config_file', default=None)

    config_args, _ = cfg_parser.parse_known_args()

    assert config_args.config is not None and config_args.config_file is not None
    with open(config_args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)[config_args.config]['train']

     # Main args
    general = parser.add_argument_group('general setup')
    general.add_argument('--work_dir', required=True, type=str,
                         help='Directory for the results')

    dataset = parser.add_argument_group('dataset setup')
    dataset.add_argument('--dataset_name', type=str, help='Name of dataset on huggingface')
    dataset.add_argument('--language', type=str, help='Language')
    dataset.add_argument('--joint_input', type=bool, help='Whether to encode muliple inputs as a single sequence')

    model = parser.add_argument_group('model setup')
    model.add_argument('--n_labels', type=int, default=3,
                       help='Number of labels')
    model.add_argument('--pretrained_path', type=str,
                       help='Path to the pretrained model')
    model.add_argument('--model_type', type=str,
                       help='If model is fixed or routed')

    opt = parser.add_argument_group('optimizer setup')
    opt.add_argument('--optim', default='adam', type=str, choices=['adam'],
                     help='Optimizer to use')
    opt.add_argument('--lr', type=float,
                     help='Initial learning rate')
    opt.add_argument('--scheduler', default='cosine', type=str,
                     choices=['cosine'], help='LR scheduler to use')
    opt.add_argument('--clip', type=float, default=0.25,
                     help='Gradient clipping')
    opt.add_argument('--weight_decay', type=float, default=0.0,
                     help='Weight decay for adam')
    opt.add_argument('--adam_b1', type=float, default=0.9)
    opt.add_argument('--adam_b2', type=float, default=0.999)
    opt.add_argument('--adam_eps', type=float, default=1e-8)

    training = parser.add_argument_group('training setup')
    training.add_argument('--max_train_steps', type=int, default=None,
                          help='Max number of training steps')
    training.add_argument('--batch_size', type=int, default=64,
                          help='Global batch size')
    training.add_argument('--seed', type=int, default=42,
                          help='Random seed')
    training.add_argument('--seq_len', type=int, default=512,
                          help='Maximum sequence length')
    training.add_argument('--report_to', type=str, default="wandb",
                          help='Wandb')
    training.add_argument('--gradient_accumulation_steps', type=int, default=1,
                          help='gradient_accumulation_steps')
    training.add_argument('--num_warmup_steps', type=int, default=5000,
                          help='num_warmup_steps')
    training.add_argument('--warmup_ratio', type=int, default=0.1,
                          help='warmup_ratio')
    training.add_argument('--logging_steps', type=int, default=500,
                          help='logging_steps')
    training.add_argument('--checkpointing_steps', type=str, help='whether to group texts ?', default="4000")
    training.add_argument('--with_tracking', type=bool, help='whether to track with wandb ?', default=True)
    training.add_argument('--resume_from_checkpoint', type=bool, help='resume_from_checkpoint', default=False)
    training.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")

    parser.set_defaults(**config)

    args, _ = parser.parse_known_args()

    return args

def main():
    args = parse_args()
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Create output directory with timestamp
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    basename = f"{os.path.basename(args.work_dir)}_{current_time}"
    new_path = os.path.join(os.path.dirname(args.work_dir), basename)
    args.output_dir = new_path
    os.makedirs(args.output_dir, exist_ok=True)

    # Accelerate config
    accelerator_log_kwargs = {}
    accelerator_log_kwargs["log_with"] = args.report_to
    accelerator_log_kwargs["project_dir"] = args.output_dir
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs], **accelerator_log_kwargs)


    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    transformers.logging.set_verbosity_error()

    set_seed(args.seed)

    # Load pretrained model
    logger.info("Loading pretrained model ....")
    pretrained_model, pretrained_config = load_pretrained_model(args.pretrained_path, args.model_type)

    boundary_kwargs = {
        'boundaries_type': pretrained_config["boundaries_type"],
        'fixed_sf': pretrained_config["fixed_sf"],
        'tokenizer_path': pretrained_config["tokenizer_path"],
        "script_tokens": pretrained_config["script_tokens"]
    }
    vocab = MixtureByteVocab(**boundary_kwargs)

    id_to_script = {value: key for key, value in pretrained_config["script_to_id"].items()}
    language_to_script_id = {lang: int(id_to_script[script]) for lang, script in  pretrained_config["language_to_script"].items()}
    logger.info(f"language_to_script_id is {language_to_script_id}")


    ###########################################################################
    # Load data
    ###########################################################################
    logger.info("Loading data corpus ....")

    data_corpus = JointInputcorpus(language=args.language,
                    dataset_name=args.dataset_name,
                    tokenizer=vocab.tokenizer,
                    max_seq_length=args.seq_len,
                    accelerator=accelerator,
                    cache_dir="cache",
                    model_type=args.model_type,
                    language_to_script_id=language_to_script_id)

    # Save config file
    save_args_to_json(args, args.output_dir)

    data_collator = default_data_collator
    train_dataloader = DataLoader(data_corpus.train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(data_corpus.validation_dataset, collate_fn=data_collator, batch_size=args.batch_size)
    test_dataloader = DataLoader(data_corpus.test_dataset, collate_fn=data_collator, batch_size=args.batch_size)

    # Initialize Classification model
    model = MagnetAverageSingleInputWithPadding(data_corpus.num_labels, pretrained_model)
    logger.info(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(args.adam_b1, args.adam_b2),
                           eps=args.adam_eps,
                           weight_decay=args.weight_decay)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    num_warmup_steps = int(args.max_train_steps * args.warmup_ratio)

    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_train_steps)

     # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader, scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["scheduler"]#.value
        accelerator.init_trackers(project_name="gradient-based-tokenization", config=experiment_config, init_kwargs={"wandb": {"entity": "owos", "name":basename}})

    # Get the metric function
    if args.dataset_name == "xnli":
        metric = evaluate.load("xnli", cache_dir="cache", experiment_id=f"{basename}_xnli")
    elif args.dataset_name == "paws-x":
        metric = evaluate.load("accuracy", cache_dir="cache", experiment_id=f"{basename}_acc")

    # Train!
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(data_corpus.train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    completed_steps = 0
    starting_epoch = 0

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            if args.joint_input:
                classification_loss, _, stats, boundary_loss = model(batch)
                boundary_loss = boundary_loss[0]
                loss = classification_loss + boundary_loss

            else:
                loss, _, stats = model(batch["x_ids"], batch["y_ids"], batch["labels"])


            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            # Gradient Clipping
            accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.clip,
                            )

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if step % args.logging_steps== 0:
                if args.with_tracking:
                    #accelerator.log({"train_loss": loss}, step=completed_steps,)
                    accelerator.log({"train_cls_loss": classification_loss, "train_loss": loss,  "train_boundary_loss": boundary_loss}, step=completed_steps)
                    accelerator.log(
                        stats,
                        step=completed_steps,
                    )
                    logger.info(f"stats are {stats}")

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        ##########################################
            # Evaluate on validation set
        ##########################################
        logger.info(f"Evaluating validation set for epoch {epoch}")
        model.eval()
        val_losses = []
        val_stats_agg = defaultdict(list)
        samples_seen = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                if args.joint_input:
                    #val_loss, val_logits, val_stats = model(batch["input_ids"], batch["labels"])
                    val_loss, val_logits, val_stats, _ = model(batch)
                else:
                    val_loss, val_logits, val_stats = model(batch["x_ids"], batch["y_ids"], batch["labels"])

            val_losses.append(accelerator.gather_for_metrics(val_loss.repeat(args.batch_size)))

            predictions = val_logits.argmax(dim=-1)
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]

            metric.add_batch(
                predictions=predictions,
                references=references)

            for k, v in val_stats.items():
                val_stats_agg[f"val_{k}"].append(v)

        # Compute boundary stats
        val_stats_mean_dict = calculate_mean(val_stats_agg)

        val_losses = torch.cat(val_losses)
        eval_loss = torch.mean(val_losses)


        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: valid {eval_metric} valid loss {eval_loss}")

        metrics_dict = {
                    "valid_accuracy" : eval_metric,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "eval_loss": eval_loss.item(),
                    "epoch": epoch,
                    "step": completed_steps,
                }
        metrics_dict.update(val_stats_mean_dict)

        if args.with_tracking:
            accelerator.log(
                metrics_dict,
                step=completed_steps,
            )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    ##########################################
            # Evaluate on the test set
    ##########################################
    logger.info("Evaluating test set")
    if args.dataset_name == "xnli":
        test_metric = evaluate.load("xnli", cache_dir="cache", experiment_id=f"{basename}_xnli")
    elif args.dataset_name == "paws-x":
        test_metric = evaluate.load("accuracy", cache_dir="cache", experiment_id=f"{basename}_acc")


    model.eval()
    test_losses = []
    test_stats_agg = defaultdict(list)
    samples_seen = 0

    all_predictions = []
    all_targets = []
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            if args.joint_input:
                #test_loss, test_logits, test_stats = model(batch["input_ids"], batch["labels"])
                test_loss, test_logits, test_stats, _ = model(batch)
            else:
                test_loss, test_logits, test_stats = model(batch["x_ids"], batch["y_ids"], batch["labels"])

        test_losses.append(accelerator.gather_for_metrics(test_loss.repeat(args.batch_size)))

        predictions = test_logits.argmax(dim=-1)
        predictions, references = accelerator.gather((predictions, batch["labels"]))
        if accelerator.num_processes > 1:
            if step == len(test_dataloader) - 1:
                predictions = predictions[: len(test_dataloader.dataset) - samples_seen]
                references = references[: len(test_dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]

        # Save predictions
        all_predictions.extend(predictions)
        all_targets.extend(references)


        test_metric.add_batch(
                predictions=predictions,
                references=references)

        for k, v in test_stats.items():
            test_stats_agg[f"test_{k}"].append(v)

    # Compute boundary stats
    test_stats_mean_dict = calculate_mean(test_stats_agg)

    test_losses = torch.cat(test_losses)
    final_test_loss = torch.mean(test_losses)

    test_metric = test_metric.compute()
    logger.info(f"epoch {epoch}: test {test_metric} test loss {final_test_loss}")

    test_metrics_dict = {
                    "test_accuracy" : test_metric,
                    "test_loss": final_test_loss,
                    "epoch": epoch,
                    "step": completed_steps,
                }

    test_metrics_dict.update(test_stats_mean_dict)

    if args.with_tracking:
            accelerator.log(
                test_metrics_dict,
                step=completed_steps,
            )

    if args.with_tracking:
        accelerator.end_training()


    final_metrics_dict = { "test_accuracy" : test_metric,
                     "valid_accuracy" : eval_metric,
                     "train_loss":  total_loss.item() / len(train_dataloader),
                     "valid_loss": eval_loss.item(),
                     "test_loss": final_test_loss.item()
                    }


    # Save Test predictions
    output_predict_file = os.path.join(args.output_dir, "predict_results.txt")
    with open(output_predict_file, "w") as writer:
        logger.info("***** Predict results *****")
        writer.write("index\tprediction\treference\n")
        for index, (pred, targ) in enumerate(zip(all_predictions, all_targets)):
            writer.write(f"{index}\t{pred}\t{targ}\n")

    logger.info("Predict results saved at {}".format(output_predict_file))

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save({
                "model": unwrapped_model.state_dict(),
                "optimizer": optimizer.optimizer.state_dict() # optimizer is an AcceleratedOptimizer object
            }, os.path.join(args.output_dir, "model.pth"))

        # save results into a json file
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                    json.dump(final_metrics_dict, f)

if __name__ == "__main__":
    main()






