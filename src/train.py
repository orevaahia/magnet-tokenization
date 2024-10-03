import argparse
import functools
import math
import os
import time
import numpy as np
import torch
import torch.optim as optim
import yaml
import inspect
import utils
import json
import logging
import transformers
import wandb

from datetime import datetime
from collections import defaultdict
from accelerate.logging import get_logger
from transformers import DataCollatorForLanguageModeling, get_scheduler
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from data_utils import MagnetDataset, MixtureByteVocab
from evaluation import evaluate_inidiv_dataset_LM
from magnet import MagnetTransformerLM
from utils import init_seed, calculate_mean, save_args_to_json

np.set_printoptions(suppress=True)
logger = get_logger(__name__)

def list_of_strings(arg):
    return arg.split(',')

def list_of_ints(arg):
    return list(map(float, arg.split(',')))

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
    general.add_argument('--work_dir', default='LM-TFM', type=str,
                         help='Directory for the results')

    dataset = parser.add_argument_group('dataset setup')
    dataset.add_argument('--data', type=str, help='Location of the data corpus')

    model = parser.add_argument_group('model setup')
    model.add_argument('--n_head', type=int, default=8,
                       help='Number of heads')
    model.add_argument('--d_head', type=int, default=64,
                       help='Head dimension')
    model.add_argument('--d_model', type=int, default=512,
                       help='Model dimension')
    model.add_argument('--d_inner', type=int, default=2048,
                       help='Inner dimension in feedforward layer')
    model.add_argument('--dropout', type=float, default=0.1,
                       help='Global dropout rate')
    model.add_argument('--dropatt', type=float, default=0.0,
                       help='Attention probability dropout rate')
    model.add_argument('--pre_lnorm', action='store_true',
                       help='Apply LayerNorm to the input instead of the output')
    model.add_argument('--init', default='normal', type=str,
                       help='Parameter initializer to use')
    model.add_argument('--emb_init', default='normal', type=str,
                       help='Parameter initializer to use')
    model.add_argument('--init_range', type=float, default=0.1,
                       help='Parameters initialized by U(-init_range, init_range)')
    model.add_argument('--emb_init_range', type=float, default=0.01,
                       help='Parameters initialized by U(-init_range, init_range)')
    model.add_argument('--init_std', type=float, default=0.02,
                       help='Parameters initialized by N(0, init_std)')
    model.add_argument('--proj_init_std', type=float, default=0.01,
                       help='Parameters initialized by N(0, init_std)')
    model.add_argument('--model_config', type=str, default="[3, (8,) ,3]",
                       help="[pre_layers, (shortened_layers, ), post_layers]")
    model.add_argument('--activation_function', type=str, default='relu')

    boundaries = parser.add_argument_group('boundary creator')
    boundaries.add_argument('--boundaries_type', type=str)
    boundaries.add_argument('--bsp_data', type=str)
    boundaries.add_argument('--tokenizer_path', type=str)
    boundaries.add_argument('--fixed_sf', type=int)
    boundaries.add_argument('--spikes_left', type=int)
    boundaries.add_argument('--temp', type=float)
    boundaries.add_argument('--prior', type=float)
    boundaries.add_argument('--script_tokens', type=list_of_strings)
    boundaries.add_argument('--prior_list', type=list_of_ints)

    opt = parser.add_argument_group('optimizer setup')
    opt.add_argument('--optim', default='adam', type=str, choices=['adam'],
                     help='Optimizer to use')
    opt.add_argument('--lr', type=float, default=0.00025,
                     help='Initial learning rate')
    opt.add_argument('--scheduler', default='cosine', type=str,
                     choices=['cosine'], help='LR scheduler to use')
    opt.add_argument('--warmup_step', type=int, default=1000,
                     help='Number of iterations for LR warmup')
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
                          help='Number of gradient accumulation steps')
    training.add_argument('--num_warmup_steps', type=int, default=5000,
                          help='Number of warm up steps')
    training.add_argument('--logging_steps', type=int, default=500,
                          help='Number of logging steps')
    training.add_argument('--line_by_line', type=bool, help='whether to group texts ?')
    training.add_argument('--checkpointing_steps', type=str, help='Checkpointing steps', default="4000")
    training.add_argument('--with_tracking', type=bool, help='whether to track with wandb ?', default=True)
    training.add_argument('--resume_from_checkpoint', type=bool, help='Whether to resume training from a checkpoint', default=False)
    training.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs.")

    val = parser.add_argument_group('validation setup')
    val.add_argument('--eval_max_steps', type=int, default=-1,
                     help='Max eval steps')
    val.add_argument('--ckpt_path', type=str, default="")

    parser.set_defaults(**config)
    args, _ = parser.parse_known_args()

    args.ckpt_path = '/'.join(config_args.config_file.split('/')[:-1])
    args.eval_batch_size = int(args.batch_size / 2)

    assert args.boundaries_type in [
        "none",
        "fixed",
        "whitespaces",
        "unigram",
        "entropy",
        "gumbel",
    ]

    return args


def main():
    args = parse_args()

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
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)


    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()


    init_seed(args.seed)

    ###########################################################################
    # Load data
    ###########################################################################
    boundary_kwargs = {
        'boundaries_type': args.boundaries_type,
        'fixed_sf': args.fixed_sf,
        'tokenizer_path': args.tokenizer_path,
        'script_tokens': args.script_tokens
    }
    logging.info("----------------------------------------------------------------")
    logging.info("----------------------------------------------------------------")
    logging.info("Preparing corpus")


    # Start the conditioning here


    # Create byte vocabulary and map scripts to their respective input ids. This is necessary for routing tokens to their boundary predictors
    vocab = MixtureByteVocab(**boundary_kwargs)
    args.script_to_id = {vocab.tokenizer.convert_tokens_to_ids(i): i  for i in args.script_tokens}
    id_to_script = {value: key for key, value in args.script_to_id.items()}
    language_to_script_id = {lang: id_to_script[script] for lang, script in  args.language_to_script.items()}
    args.all_script_ids_dict = {j:i for i,j in zip(args.prior_list, args.script_tokens)}


    print(f"after language_to_script_id is {language_to_script_id}")

    # Load dataset
    data_corpus = MagnetDataset(args.data, args.seq_len, accelerator, language_to_script_id, **boundary_kwargs)
    # n_token here determines the model embedding size
    args.n_token = len(vocab)
    logging.info(f"Script ids dict is  {args.all_script_ids_dict}")

    # Save config file
    save_args_to_json(args, args.output_dir)

    # Dataloader and data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=vocab.tokenizer, mlm=False, return_tensors="pt")
    train_dataloader = DataLoader(data_corpus.train_dataset, collate_fn=data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(data_corpus.validation_dataset, collate_fn=data_collator, batch_size=args.batch_size)
    test_dataloader = DataLoader(data_corpus.test_dataset, collate_fn=data_collator, batch_size=args.batch_size)

    # Log a few random samples from the training set:
    #for index in random.sample(range(len(data_corpus.train_dataset)), 3):
        #logger.info(f"Sample {index} of the training set: {data_corpus.train_dataset[index]}.")

    def get_model_config():
        model_args = inspect.getfullargspec(MagnetTransformerLM).args
        assert model_args.index('self') == 0
        model_args = model_args[1:]
        return {arg: getattr(args, arg) for arg in model_args}

    ###########################################################################
    # Build model
    ###########################################################################

    model = MagnetTransformerLM(**get_model_config())
    model.apply(functools.partial(utils.weights_init, args=args))
    model.word_emb.apply(functools.partial(utils.weights_init, args=args))
    args.n_all_param = sum([p.nelement() for p in model.parameters()])
    logger.info(model)

    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(args.adam_b1, args.adam_b2),
                           eps=args.adam_eps,
                           weight_decay=args.weight_decay)

    overrode_max_train_steps = False

    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, scheduler
    )

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["scheduler"]#.value


    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    ###########################################################################
    # Start Training
    ###########################################################################

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            #sth wrong here
            #completed_steps = starting_epoch * num_update_steps_per_epoch
            completed_steps = starting_epoch #* num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            # total_train_aux_loss is the boundary predictor loss
            total_train_loss, total_train_aux_loss, total_count = 0, 0, 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        train_stats_agg = defaultdict(list)
        for step, batch in enumerate(tqdm(active_dataloader)):
            with accelerator.accumulate(model):
                inputs, target  = batch["input_ids"], batch["labels"]
                seq_loss, stats, aux_loss,  _ = model(inputs, target, "LM")

                # Combine auxilliary boundary predictor loss and language modelling loss
                # Sometimes you might have only one script in a batch and the auxiliary loss might be one
                if len(aux_loss) > 1:
                    #loss = seq_loss + torch.stack(aux_loss).sum()
                    loss = seq_loss + torch.mean(torch.stack(aux_loss))
                else:
                    loss = seq_loss + aux_loss

                if args.with_tracking:
                    total_train_loss += seq_loss.detach().float()
                    if len(aux_loss) > 1:
                        total_train_aux_loss += torch.mean(torch.stack(aux_loss)).detach().float()
                        #total_train_aux_loss += torch.stack(aux_loss).sum().detach().float()
                    else:
                         total_train_aux_loss += aux_loss.detach().float()
                    total_count += 1

                    for k, v in stats.items():
                        train_stats_agg[f"train_{k}"].append(v)

                # Gradient Clipping
                #torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if step % args.logging_steps== 0:
                if args.with_tracking:
                    accelerator.log(
                        {"train_lm_loss": seq_loss,
                        "train_bpc": loss / math.log(2)},
                        step=completed_steps,
                    )
                    accelerator.log(
                        stats,
                        step=completed_steps,
                    )

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= args.max_train_steps:
                break

        # take the mean afer every epoch
        train_stats_mean_dict = calculate_mean(train_stats_agg)

        ###########################################################################
        # Evaluate Validation Set after each epoch
        ###########################################################################
        model.eval()
        losses, val_aux_losses = [], []
        val_stats_agg = defaultdict(list)

        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                inputs, target  = batch["input_ids"], batch["labels"]
                loss, val_stats, val_aux_loss, _ = model(inputs, target, "LM")

            losses.append(accelerator.gather_for_metrics(loss.repeat(args.batch_size)))

            if len(val_aux_loss) > 1:
                val_aux_losses.append(accelerator.gather_for_metrics(torch.mean(torch.stack(val_aux_loss)).repeat(args.batch_size)))
            else:
                val_aux_losses.append(accelerator.gather_for_metrics(val_aux_loss[0].repeat(args.batch_size)))

            for k, v in val_stats.items():
                val_stats_agg[f"val_{k}"].append(v)


        val_stats_mean_dict = calculate_mean(val_stats_agg)

        losses = torch.cat(losses)
        val_aux_losses = torch.cat(val_aux_losses)

        try:
            eval_loss = torch.mean(losses)
            eval_val_aux_loss = torch.mean(val_aux_losses)
            eval_bpc = eval_loss / math.log(2)
        except OverflowError:
            eval_bpc = float("inf")

        train_loss = total_train_loss.item() / total_count
        train_aux_loss = total_train_aux_loss.item() / total_count

        logger.info(f"epoch {epoch}: train_bpc: {train_loss / math.log(2)}  train_loss: {train_loss}  train_aux_loss: {train_aux_loss} eval_bpc: {eval_bpc} eval_loss: {eval_loss}  eval_aux_loss: {eval_val_aux_loss} ")

        # Save all metrics into a dictionary
        metrics_dict =  {
                    "train_bpc": train_loss / math.log(2),
                    "train_lm_loss": train_loss,
                    "eval_bpc": eval_bpc.item() ,
                    "eval_lm_loss": eval_loss.item(),
                    "train_aux_loss": train_aux_loss,
                    "eval_aux_loss": eval_val_aux_loss.item(),
                    "epoch": epoch,
                    "step": completed_steps,
                }

        # Update the metrics dict
        metrics_dict.update(train_stats_mean_dict)
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

    # Evaluate individual languages
    logger.info("Start evaluating individual languages")
    languages_bpc_dictionary, languages_loss_dictionary = evaluate_inidiv_dataset_LM(data_corpus.individual_validation_dataset,
                            data_collator,
                            args.batch_size,
                            accelerator,
                            model)

    languages_bpc_dictionary.update(languages_loss_dictionary)

    # log languages dict
    if args.with_tracking:
        accelerator.log(
                    languages_bpc_dictionary,
                    step=completed_steps,
                )

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        accelerator.save({
                "model": unwrapped_model.state_dict(),
                "optimizer": optimizer.optimizer.state_dict() # optimizer is an AcceleratedOptimizer object
            }, os.path.join(args.output_dir, "model.pth"))

        #accelerator.save_model(model, args.output_dir)

        if accelerator.is_main_process:
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(metrics_dict, f)

            with open(os.path.join(args.output_dir, "language_eval_results.json"), "w") as f:
                json.dump(languages_bpc_dictionary, f)


if __name__ == "__main__":
    main()

