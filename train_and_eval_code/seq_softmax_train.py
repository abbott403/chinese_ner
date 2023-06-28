import torch
from torch import nn
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, LinearLR
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from transformers import BertTokenizerFast, BertConfig
import os
from tqdm import tqdm
from sklearn.metrics import classification_report
import time

from models.softmax_ner import BertSoftmax
from utils.all_loss import get_loss_function
from utils.all_metrics import SeqEntityScore
from train_config import seq_config as configs
from utils.utils import set_random_seed, ddp_reduce_mean, freeze_weight
from data_process.seq_dataloader import data_generator, data_generator_ddp
from callback.adversarial import FGM


def train(model, dataloader, epoch, optimizer, scheduler, device, loss_fn):
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
        active_loss = batch_attention_mask.view(-1) == 1
        active_logits = output.view(-1, len(configs.ent2id))[active_loss]
        active_labels = batch_labels.view(-1)[active_loss]
        loss = loss_fn(active_logits, active_labels)
        # loss = LOSS_FUNC_LIST[configs.loss_type](logits.view(-1, self.num_labels), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)

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

    flat_prediction = sum(prediction_symbol, [])
    flat_label = sum(label_symbol, [])
    print(classification_report(flat_label, flat_prediction))
    metrics.update(label_symbol, prediction_symbol)
    eval_info, entity_info = metrics.result()

    print("******************************************")
    print(eval_info)
    print("******************************************")
    print(entity_info)
    return eval_info["f1"]


def main():
    set_random_seed(configs.seed)
    ent_type_size = len(configs.ent2id)
    os.environ["TOKENIZERS_PARALLELISM"] = 'true'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    local_time = time.localtime(time.time())
    time_str = time.strftime("%H-%M-%S", local_time)
    output_writer = SummaryWriter(f"train_logs/softmax/{time_str}")
    
    tokenizer = BertTokenizerFast.from_pretrained(configs.pretrained_model_path, add_special_tokens=True,
                                                  do_lower_case=False)
    train_dataloader, valid_dataloader = data_generator(tokenizer)

    bert_config = BertConfig.from_pretrained(configs.pretrained_model_path)
    model = BertSoftmax(bert_config, ent_type_size, configs.dropout_rate)
    unfreeze_layer = ["layer.10", "layer.11", "classifier."]
    freeze_weight(model, unfreeze_layer)
    model = model.to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=configs.softmax_learning_rate)
    label_weight = torch.tensor([1, 10, 10, 10, 10, 10, 10], dtype=torch.float).to(device)
    loss_fn = get_loss_function(label_weight, configs.loss_type)

    if configs.scheduler == "CAWR":
        t_mult = configs.cawr_scheduler["T_mult"]
        rewarm_epoch_num = configs.cawr_scheduler["rewarm_epoch_num"]
        scheduler = CosineAnnealingWarmRestarts(optimizer, len(train_dataloader) * rewarm_epoch_num, t_mult)
    elif configs.scheduler == "Step":
        decay_rate = configs.step_scheduler["decay_rate"]
        decay_steps = configs.step_scheduler["decay_steps"]
        scheduler = StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)
    elif configs.scheduler == "Linear":
        scheduler = LinearLR(optimizer, 1, 0.1, configs.num_train_epoch * len(train_dataloader))
    else:
        scheduler = None
    metrics = SeqEntityScore()

    max_f1 = 0.
    for epoch in range(configs.num_train_epoch):
        loss = train(model, train_dataloader, epoch, optimizer, scheduler, device, loss_fn)
        valid_f1 = valid(model, valid_dataloader, metrics, device)
        output_writer.add_scalar("train_loss", loss, epoch)
        output_writer.add_scalar("valid_f1", valid_f1, epoch)
        if valid_f1 > max_f1:
            max_f1 = valid_f1
            if max_f1 > configs.f1_save_threshold:
                model_f1_val = int(round(max_f1, 3) * 1000)
                torch.save(model.state_dict(), os.path.join(configs.model_save_path, "so_{}.pt".format(model_f1_val)))

        print(f"Best F1: {max_f1}")


def train_ddp(model, dataloader, optimizer, scheduler, device, adversarial, amp_scaler, loss_fn):
    model.train()

    total_loss = 0.0
    avg_loss = 0.0
    for batch_id, batch_data in enumerate(dataloader):
        optimizer.zero_grad()

        batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_data
        batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = (batch_input_ids.to(device),
                                                                                     batch_attention_mask.to(device),
                                                                                     batch_token_type_ids.to(device),
                                                                                     batch_labels.to(device))
        active_loss = batch_attention_mask.view(-1) == 1

        if configs.use_amp:
            with autocast():
                logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
                active_logits = logits.view(-1, len(configs.ent2id))[active_loss]
                active_labels = batch_labels.view(-1)[active_loss]
                loss = loss_fn(active_logits, active_labels)
            dist.barrier()
            amp_scaler.scale(loss).backward()

            if configs.use_attack:
                adversarial.attack()
                with autocast():
                    logits_adv = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
                    active_logits = logits_adv.view(-1, len(configs.ent2id))[active_loss]
                    active_labels = batch_labels.view(-1)[active_loss]
                    loss_dev = loss_fn(active_logits, active_labels)
                dist.barrier()
                amp_scaler.scale(loss_dev).backward()
                adversarial.restore()

            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            active_logits = logits.view(-1, len(configs.ent2id))[active_loss]
            active_labels = batch_labels.view(-1)[active_loss]
            loss = loss_fn(active_logits, active_labels)
            dist.barrier()
            loss.backward()

            if configs.use_attack:
                adversarial.attack()  # 在embedding上添加对抗扰动
                logits_adv = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
                active_logits = logits_adv.view(-1, len(configs.ent2id))[active_loss]
                active_labels = batch_labels.view(-1)[active_loss]
                loss_dev = loss_fn(active_logits, active_labels)
                dist.barrier()

                loss_dev.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                adversarial.restore()  # 恢复embedding参数

            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        all_reduce_loss = ddp_reduce_mean(loss, configs.nprocs_per_node)
        total_loss += all_reduce_loss.item()
        avg_loss = total_loss / (batch_id + 1)

    return avg_loss


def valid_ddp(model, dataloader, metrics, device):
    model.eval()
    metrics.reset()

    label_symbol, prediction_symbol = [], []
    for batch_data in tqdm(dataloader):
        batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_data
        batch_input_ids, batch_attention_mask, batch_token_type_ids = (batch_input_ids.to(device),
                                                                       batch_attention_mask.to(device),
                                                                       batch_token_type_ids.to(device))
        with torch.no_grad():
            logits = model.module(batch_input_ids, batch_attention_mask, batch_token_type_ids)

        predictions = logits.argmax(dim=-1).cpu().numpy().tolist()
        labels = batch_labels.numpy().tolist()

        prediction_symbol += [[configs.id2ent[int(p)] for (p, l) in zip(prediction, label) if l != -100]
                              for prediction, label in zip(predictions, labels)]
        label_symbol += [[configs.id2ent[int(l)] for l in label if l != -100] for label in labels]

    flat_prediction = sum(prediction_symbol, [])
    flat_label = sum(label_symbol, [])
    print(classification_report(flat_label, flat_prediction))
    metrics.update(label_symbol, prediction_symbol)
    eval_info, entity_info = metrics.result()

    print("******************************************")
    print(eval_info)
    print("******************************************")
    print(entity_info)
    return eval_info["f1"]


def main_ddp():
    # args = get_parse_args()
    set_random_seed(configs.seed)
    ent_type_size = len(configs.ent2id)

    os.environ["TOKENIZERS_PARALLELISM"] = 'true'

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if local_rank == 0:
        local_time = time.localtime(time.time())
        time_str = time.strftime("%H-%M-%S", local_time)
        output_writer = SummaryWriter(f"train_logs/softmax/{time_str}")

    tokenizer = BertTokenizerFast.from_pretrained(configs.pretrained_model_path, add_special_tokens=True,
                                                  do_lower_case=False)
    train_dataloader, valid_dataloader, train_sampler = data_generator_ddp(tokenizer)

    bert_config = BertConfig.from_pretrained(configs.pretrained_model_path)
    model = BertSoftmax(bert_config, ent_type_size, configs.dropout_rate)
    unfreeze_layer = ["layer.10", "layer.11", "classifier."]
    freeze_weight(model, unfreeze_layer)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    fgm = FGM(model, epsilon=1) if configs.use_attack else None
    scaler = GradScaler() if configs.use_amp else None
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=configs.softmax_learning_rate)
    label_weight = torch.tensor([1, 10, 10, 10, 10, 10, 10], dtype=torch.float).to(device)
    loss_fn = get_loss_function(label_weight, configs.loss_type)

    if configs.scheduler == "CAWR":
        T_mult = configs.cawr_scheduler["T_mult"]
        rewarm_epoch_num = configs.cawr_scheduler["rewarm_epoch_num"]
        scheduler = CosineAnnealingWarmRestarts(optimizer, len(train_dataloader) * rewarm_epoch_num, T_mult)
    elif configs.scheduler == "Step":
        decay_rate = configs.step_scheduler["decay_rate"]
        decay_steps = configs.step_scheduler["decay_steps"]
        scheduler = StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)
    elif configs.scheduler == "Linear":
        scheduler = LinearLR(optimizer, 1, 0.1, configs.num_train_epoch * len(train_dataloader))
    else:
        scheduler = None
    metrics = SeqEntityScore()

    max_f1 = 0.
    for epoch in range(configs.num_train_epoch):
        train_sampler.set_epoch(epoch)
        print("Rank:{} - Epoch {}/{}\n".format(local_rank, epoch, configs.num_train_epoch - 1))

        avg_loss = train_ddp(model, train_dataloader, optimizer, scheduler, device, fgm, scaler, loss_fn)
        if local_rank == 0:
            valid_f1 = valid_ddp(model, valid_dataloader, metrics, device)
            output_writer.add_scalar("loss", avg_loss, epoch)
            output_writer.add_scalar("f1", valid_f1, epoch)

            if valid_f1 > max_f1:
                max_f1 = valid_f1
                if max_f1 > configs.f1_save_threshold:
                    model_f1_val = int(round(max_f1, 3) * 1000)
                    torch.save(model.module.state_dict(), os.path.join(configs.model_save_path, "so_{}.pt".format(model_f1_val)))

            print(f"Best F1: {max_f1}")

        print(f"Rank:{local_rank} waiting before the barrier\n")
        dist.barrier()
        print(f"Rank:{local_rank} left the barrier\n")


if __name__ == "__main__":
    if configs.is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        main_ddp()
    else:
        main()
