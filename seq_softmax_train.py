import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from transformers import BertTokenizerFast
import os
from tqdm import tqdm

from models.softmax_ner import BertSoftmax
from utils.all_loss import FocalLoss, LabelSmoothingCrossEntropy
from utils.all_metrics import SeqEntityScore
from train_config import seq_config as configs
from utils.utils import set_random_seed
from data_process.seq_dataloader import data_generator, data_generator_ddp
from callback.adversarial import FGM


LOSS_FUNC_LIST = {"cross_entropy": CrossEntropyLoss(),
                  "label_smooth": LabelSmoothingCrossEntropy(),
                  "focal": FocalLoss()}


def ddp_reduce_mean(loss_data, nprocs):
    rt = loss_data.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def train(model, dataloader, epoch, optimizer, scheduler, device):
    model.train()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    total_loss = 0.0
    avg_loss = 0.0
    for batch_id, batch_data in pbar:
        batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_data
        batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = (batch_input_ids.to(device),
                                                                                     batch_attention_mask.to(device),
                                                                                     batch_token_type_ids.to(device),
                                                                                     batch_labels.to(device))

        output = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
        # active_loss = batch_attention_mask.view(-1) == 1
        active_logits = output.view(-1, len(configs.ent2id))
        active_labels = batch_labels.view(-1)
        loss = LOSS_FUNC_LIST[configs.loss_type](active_logits, active_labels)
        # loss = LOSS_FUNC_LIST[configs.loss_type](logits.view(-1, self.num_labels), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (batch_id + 1)

        if scheduler is not None:
            scheduler.step()

        pbar.set_description(f'Epoch: {epoch + 1}/{configs.num_train_epoch}, Step: {batch_id + 1}/{len(dataloader)}')
        pbar.set_postfix(loss=avg_loss, lr=optimizer.param_groups[0]["lr"])

    return avg_loss


def valid(model, dataloader, metrics, device):
    model.eval()
    metrics.reset()

    label_symbol, prediction_symbol = [], []
    for batch_data in tqdm(dataloader):
        batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_data
        batch_input_ids, batch_attention_mask, batch_token_type_ids = (batch_input_ids.to(device),
                                                                       batch_attention_mask.to(device),
                                                                       batch_token_type_ids.to(device))
        with torch.no_grad():
            logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
        predictions = logits.argmax(dim=-1).cpu().numpy().tolist()
        labels = batch_labels.numpy().tolist()
        prediction_symbol += [[configs.id2ent[int(p)] for (p, l) in zip(prediction, label) if l != -100]
                              for prediction, label in zip(predictions, labels)]
        label_symbol += [[configs.id2ent[int(l)] for l in label if l != -100] for label in labels]

    metrics.update(label_symbol, prediction_symbol)
    eval_info, entity_info = metrics.result()

    print("******************************************")
    print(eval_info, "\n")
    print("******************************************")
    print(entity_info)
    return eval_info["f1"]


def main():
    set_random_seed(configs.seed)
    ent_type_size = len(configs.ent2id)

    os.environ["TOKENIZERS_PARALLELISM"] = 'true'

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    output_writer = SummaryWriter("train_logs/softmax/")
    tokenizer = BertTokenizerFast.from_pretrained(configs.pretrained_model_path, add_special_tokens=True,
                                                  do_lower_case=False)
    train_dataloader, valid_dataloader = data_generator(tokenizer)

    model = BertSoftmax(ent_type_size, configs.dropout_rate)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.learning_rate)
    if configs.scheduler == "CAWR":
        T_mult = configs.cawr_scheduler["T_mult"]
        rewarm_epoch_num = configs.cawr_scheduler["rewarm_epoch_num"]
        scheduler = CosineAnnealingWarmRestarts(optimizer, len(train_dataloader) * rewarm_epoch_num, T_mult)
    elif configs.scheduler == "Step":
        decay_rate = configs.step_scheduler["decay_rate"]
        decay_steps = configs.step_scheduler["decay_steps"]
        scheduler = StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)
    else:
        scheduler = None
    metrics = SeqEntityScore()

    max_f1 = 0.
    for epoch in range(configs.num_train_epoch):
        loss = train(model, train_dataloader, epoch, optimizer, scheduler, device)
        valid_f1 = valid(model, valid_dataloader, metrics, device)
        output_writer.add_scalar("train_loss", loss, epoch)
        output_writer.add_scalar("valid_f1", valid_f1, epoch)
        if valid_f1 > max_f1:
            max_f1 = valid_f1
            if max_f1 > configs.f1_save_threshold:
                model_f1_val = int(round(max_f1, 3) * 1000)
                torch.save(model.state_dict(), os.path.join(configs.model_save_path, "gp_{}.pt".format(model_f1_val)))

        print(f"Best F1: {max_f1}")


if __name__ == "__main__":
    main()