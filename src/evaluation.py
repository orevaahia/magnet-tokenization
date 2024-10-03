import math
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


def evaluate_inidiv_dataset_LM(datasets, data_collator, batch_size, accelerator, model):
    """
    Evaluate individual lanaguages
    """
    bpc_dictionary = {}
    loss_dictionary = {}
    model.eval()
    for i in datasets:
        dataset = datasets[i]
        dataloader = DataLoader(dataset,
                                collate_fn=data_collator,
                                batch_size=batch_size,
                                shuffle=False)
        dataloader = accelerator.prepare(dataloader)
        losses = []
        for step, batch in enumerate(tqdm(dataloader, desc=f'evaluating {i} language...')):
            with torch.no_grad():
                inputs, target  = batch["input_ids"], batch["labels"]
                boundaries = None
                seq_loss, stats, aux_loss, _ = model(inputs, target, "LM")

            losses.append(accelerator.gather_for_metrics(seq_loss.repeat(batch_size)))

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            eval_bpc = eval_loss / math.log(2)
        except OverflowError:
                eval_bpc = float("inf")

        bpc_dictionary[f"{i}_eval_bpc"] = eval_bpc.item()
        loss_dictionary[f"{i}_eval_loss"] = eval_loss.item()

        print(f"Finished evaluating {i} language")
    return bpc_dictionary, loss_dictionary


