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

from models.span_ner import BertSpan
from utils.all_loss import get_loss_function
from utils.all_metrics import SpanEntityScore
from train_config import span_config as configs
from utils.utils import set_random_seed, ddp_reduce_mean, span_extract_item, freeze_weight
from data_process.span_dataloader import data_generator, data_generator_ddp
from callback.adversarial import FGM


def train(model, dataloader, epoch, optimizer, scheduler, device, loss_fn):
    model.train()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    total_loss = 0.0
    avg_loss = 0.0
    for batch_id, batch_data in pbar:
        batch_input_ids, batch_attention_mask, batch_token_type_ids, start_ids, end_ids, _ = batch_data
        batch_input_ids, batch_attention_mask, batch_token_type_ids, start_ids, end_ids = \
            (batch_input_ids.to(device), batch_attention_mask.to(device),
             batch_token_type_ids.to(device), start_ids.to(device), end_ids.to(device))

        zeros = torch.zeros_like(start_ids)
        new_start_ids = torch.where(start_ids < 0, zeros, start_ids)

        start_logits, end_logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids, new_start_ids)
        start_logits = start_logits.view(-1, len(configs.ent2id))
        end_logits = end_logits.view(-1, len(configs.ent2id))
        active_loss = batch_attention_mask.view(-1) == 1
        active_start_logits = start_logits[active_loss]
        active_end_logits = end_logits[active_loss]

        active_start_labels = start_ids.view(-1)[active_loss]
        active_end_labels = end_ids.view(-1)[active_loss]

        start_loss = loss_fn(active_start_logits, active_start_labels)
        end_loss = loss_fn(active_end_logits, active_end_labels)
        loss = (start_loss + end_loss) / 2

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

    for batch_data in dataloader:
        batch_input_ids, batch_attention_mask, batch_token_type_ids, start_ids, end_ids, entity_list = batch_data
        batch_input_ids, batch_attention_mask, batch_token_type_ids, start_ids, end_ids = \
            (batch_input_ids.to(device), batch_attention_mask.to(device),
             batch_token_type_ids.to(device), start_ids.to(device), end_ids.to(device))

        with torch.no_grad():
            start_logits, end_logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)

        item_res = span_extract_item(start_logits, end_logits)
        metrics.update(entity_list, item_res)

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
    # device = torch.device("cpu")
    output_writer = SummaryWriter("train_logs/span/")
    tokenizer = BertTokenizerFast.from_pretrained(configs.pretrained_model_path, add_special_tokens=True,
                                                  do_lower_case=False)
    train_dataloader, valid_dataloader = data_generator(tokenizer)

    bert_config = BertConfig.from_pretrained(configs.pretrained_model_path)
    model = BertSpan(bert_config, ent_type_size, configs.dropout_rate, configs.soft_label)
    unfreeze_layer = ["layer.11", "start_fc.", "end_fc."]
    freeze_weight(model, unfreeze_layer)
    model = model.to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=configs.learning_rate)
    label_weight = torch.tensor([1, 50, 50, 50], dtype=torch.float).to(device)
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
    metrics = SpanEntityScore()

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
                torch.save(model.state_dict(), os.path.join(configs.model_save_path, "sp_{}.pt".format(model_f1_val)))

        print(f"Best F1: {max_f1}")


def train_ddp(model, dataloader, optimizer, scheduler, device, adversarial, amp_scaler, loss_fn):
    model.train()

    total_loss = 0.0
    avg_loss = 0.0
    for batch_id, batch_data in enumerate(dataloader):
        optimizer.zero_grad()

        batch_input_ids, batch_attention_mask, batch_token_type_ids, start_ids, end_ids, _ = batch_data
        batch_input_ids, batch_attention_mask, batch_token_type_ids, start_ids, end_ids = \
            (batch_input_ids.to(device), batch_attention_mask.to(device),
             batch_token_type_ids.to(device), start_ids.to(device), end_ids.to(device))

        zeros = torch.zeros_like(start_ids)
        new_start_ids = torch.where(start_ids < 0, zeros, start_ids)

        active_loss = batch_attention_mask.view(-1) == 1
        active_start_labels = start_ids.view(-1)[active_loss]
        active_end_labels = end_ids.view(-1)[active_loss]

        if configs.use_amp:
            with autocast():
                start_logits, end_logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids,
                                                 new_start_ids)
                start_logits = start_logits.view(-1, len(configs.ent2id))
                end_logits = end_logits.view(-1, len(configs.ent2id))

                active_start_logits = start_logits[active_loss]
                active_end_logits = end_logits[active_loss]

                start_loss = loss_fn(active_start_logits, active_start_labels)
                end_loss = loss_fn(active_end_logits, active_end_labels)
                loss = (start_loss + end_loss) / 2

            dist.barrier()
            amp_scaler.scale(loss).backward()

            if configs.use_attack:
                adversarial.attack()
                with autocast():
                    start_logits_dev, end_logits_dev = model(batch_input_ids, batch_attention_mask,
                                                             batch_token_type_ids,
                                                             new_start_ids)
                    start_logits_dev = start_logits_dev.view(-1, len(configs.ent2id))
                    end_logits_dev = end_logits_dev.view(-1, len(configs.ent2id))

                    active_start_logits = start_logits_dev[active_loss]
                    active_end_logits = end_logits_dev[active_loss]

                    start_loss_dev = loss_fn(active_start_logits, active_start_labels)
                    end_loss_dev = loss_fn(active_end_logits, active_end_labels)
                    loss_dev = (start_loss_dev + end_loss_dev) / 2

                dist.barrier()
                amp_scaler.scale(loss_dev).backward()
                adversarial.restore()

            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            start_logits, end_logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids, new_start_ids)
            start_logits = start_logits.view(-1, len(configs.ent2id))
            end_logits = end_logits.view(-1, len(configs.ent2id))

            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            start_loss = loss_fn(active_start_logits, active_start_labels)
            end_loss = loss_fn(active_end_logits, active_end_labels)
            loss = (start_loss + end_loss) / 2

            dist.barrier()
            loss.backward()

            if configs.use_attack:
                adversarial.attack()  # 在embedding上添加对抗扰动
                start_logits_dev, end_logits_dev = model(batch_input_ids, batch_attention_mask,
                                                         batch_token_type_ids,
                                                         new_start_ids)
                start_logits_dev = start_logits_dev.view(-1, len(configs.ent2id))
                end_logits_dev = end_logits_dev.view(-1, len(configs.ent2id))

                active_start_logits = start_logits_dev[active_loss]
                active_end_logits = end_logits_dev[active_loss]

                start_loss_dev = loss_fn(active_start_logits, active_start_labels)
                end_loss_dev = loss_fn(active_end_logits, active_end_labels)
                loss_dev = (start_loss_dev + end_loss_dev) / 2

                dist.barrier()

                loss_dev.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                adversarial.restore()  # 恢复embedding参数

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

    for batch_data in tqdm(dataloader):
        batch_input_ids, batch_attention_mask, batch_token_type_ids, start_ids, end_ids, entity_list = batch_data
        batch_input_ids, batch_attention_mask, batch_token_type_ids, start_ids, end_ids = \
            (batch_input_ids.to(device), batch_attention_mask.to(device),
             batch_token_type_ids.to(device), start_ids.to(device), end_ids.to(device))

        with torch.no_grad():
            start_logits, end_logits = model.module(batch_input_ids, batch_attention_mask, batch_token_type_ids)

        item_res = span_extract_item(start_logits, end_logits)
        metrics.update(entity_list, item_res)

    eval_info, entity_info = metrics.result()

    print("******************************************")
    print(eval_info, "\n")
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
        output_writer = SummaryWriter("train_logs/span/")

    tokenizer = BertTokenizerFast.from_pretrained(configs.pretrained_model_path, add_special_tokens=True,
                                                  do_lower_case=False)
    train_dataloader, valid_dataloader, train_sampler = data_generator_ddp(tokenizer)

    bert_config = BertConfig.from_pretrained(configs.pretrained_model_path)
    model = BertSpan(bert_config, ent_type_size, configs.dropout_rate, configs.soft_label)
    unfreeze_layer = ["layer.11", "start_fc.", "end_fc."]
    freeze_weight(model, unfreeze_layer)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    fgm = FGM(model, epsilon=1) if configs.use_attack else None
    scaler = GradScaler() if configs.use_amp else None

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=configs.learning_rate)
    label_weight = torch.tensor([1, 50, 50, 50], dtype=torch.float).to(device)
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
    metrics = SpanEntityScore()

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
                    torch.save(model.module.state_dict(),
                               os.path.join(configs.model_save_path, "sp_{}.pt".format(model_f1_val)))

            print(f"Best F1: {max_f1} \n")

        print(f"Rank:{local_rank} waiting before the barrier")
        dist.barrier()
        print(f"Rank:{local_rank} left the barrier")


if __name__ == "__main__":
    if configs.is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        main_ddp()
    else:
        main()
