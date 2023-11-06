import torch
from torch import nn
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, LinearLR
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from transformers import BertTokenizerFast
import os
from tqdm import tqdm

from models.global_point import GlobalPoint
from utils.all_loss import multilabel_categorical_crossentropy
from utils.all_metrics import GlobalPointerScore
from configs import global_point_config as configs
from utils.utils import set_random_seed, ddp_reduce_mean
from data.global_point_dataloader import data_generator, data_generator_ddp
from utils.adversarial import FGM


def train(model, dataloader, epoch, optimizer, scheduler, device):
    model.train()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    total_loss = 0.0
    avg_loss = 0.0
    for batch_id, batch_data in pbar:
        batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_data
        batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = (batch_input_ids.to(device),
                                                                                     batch_attention_mask.to(device),
                                                                                     batch_token_type_ids.to(device),
                                                                                     batch_labels.to(device))

        logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
        loss = multilabel_categorical_crossentropy(batch_labels, logits)
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

    total_f1, total_precision, total_recall = 0., 0., 0.
    for batch_data in tqdm(dataloader):
        batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_data
        batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = (batch_input_ids.to(device),
                                                                                     batch_attention_mask.to(device),
                                                                                     batch_token_type_ids.to(device),
                                                                                     batch_labels.to(device))
        with torch.no_grad():
            logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
        sample_f1, sample_precision, sample_recall = metrics.get_evaluate_fpr(logits, batch_labels)

        total_f1 += sample_f1
        total_precision += sample_precision
        total_recall += sample_recall

    avg_f1 = total_f1 / (len(dataloader))
    avg_precision = total_precision / len(dataloader)
    avg_recall = total_recall / (len(dataloader))

    print("******************************************")
    print(f'avg_precision: {avg_precision}, avg_recall: {avg_recall}, avg_f1: {avg_f1}')
    print("******************************************")
    return avg_f1


def main():
    set_random_seed(configs.seed)
    ent_type_size = len(configs.ent2id)

    os.environ["TOKENIZERS_PARALLELISM"] = 'true'

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    output_writer = SummaryWriter("train_logs/global/")
    tokenizer = BertTokenizerFast.from_pretrained(configs.pretrained_model_path, add_special_tokens=True,
                                                  do_lower_case=False)
    train_dataloader, valid_dataloader, train_sampler = data_generator(tokenizer)

    model = GlobalPoint(ent_type_size, 64)
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
    elif configs.scheduler == "Linear":
        scheduler = LinearLR(optimizer, 1, 0.1, configs.num_train_epoch * len(train_dataloader))
    else:
        scheduler = None
    metrics = GlobalPointerScore()

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


def train_ddp(model, dataloader, optimizer, scheduler, device, adversarial, amp_scaler):
    model.train()

    total_loss = 0.0
    avg_loss = 0.0
    for batch_id, batch_data in enumerate(dataloader):
        optimizer.zero_grad()

        batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_data
        batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = (batch_input_ids.to(device),
                                                                                     batch_attention_mask.to(device),
                                                                                     batch_token_type_ids.to(device),
                                                                                     batch_labels.to(device))

        if configs.use_amp:
            with autocast():
                logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
                loss = multilabel_categorical_crossentropy(batch_labels, logits)
            dist.barrier()
            amp_scaler.scale(loss).backward()

            if configs.use_attack:
                adversarial.attack()
                with autocast():
                    logits_adv = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
                    loss_dev = multilabel_categorical_crossentropy(batch_labels, logits_adv)
                dist.barrier()
                amp_scaler.scale(loss_dev).backward()
                adversarial.restore()

            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            loss = multilabel_categorical_crossentropy(batch_labels, logits)
            dist.barrier()
            loss.backward()

            if configs.use_attack:
                adversarial.attack()  # 在embedding上添加对抗扰动
                logits_adv = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
                loss_dev = multilabel_categorical_crossentropy(batch_labels, logits_adv)
                dist.barrier()

                loss_dev.backward()       # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
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

    total_f1, total_precision, total_recall = 0., 0., 0.
    for batch_data in tqdm(dataloader):
        batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_data
        batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = (batch_input_ids.to(device),
                                                                                     batch_attention_mask.to(device),
                                                                                     batch_token_type_ids.to(device),
                                                                                     batch_labels.to(device))
        with torch.no_grad():
            logits = model.module(batch_input_ids, batch_attention_mask, batch_token_type_ids)
        sample_f1, sample_precision, sample_recall = metrics.get_evaluate_fpr(logits, batch_labels)

        total_f1 += sample_f1
        total_precision += sample_precision
        total_recall += sample_recall

    avg_f1 = total_f1 / (len(dataloader))
    avg_precision = total_precision / len(dataloader)
    avg_recall = total_recall / (len(dataloader))

    print("******************************************")
    print(f'avg_precision: {avg_precision}, avg_recall: {avg_recall}, avg_f1: {avg_f1}')
    print("******************************************")

    return avg_f1, avg_precision, avg_recall


def main_ddp():
    # args = get_parse_args()
    set_random_seed(configs.seed)
    ent_type_size = len(configs.ent2id)

    os.environ["TOKENIZERS_PARALLELISM"] = 'true'

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if local_rank == 0:
        output_writer = SummaryWriter("train_logs/")

    tokenizer = BertTokenizerFast.from_pretrained(configs.pretrained_model_path, add_special_tokens=True,
                                                  do_lower_case=False)
    train_dataloader, valid_dataloader, train_sampler = data_generator_ddp(tokenizer)

    model = GlobalPoint(ent_type_size, 64)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    fgm = FGM(model, epsilon=1) if configs.use_attack else None
    scaler = GradScaler() if configs.use_amp else None
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.learning_rate)

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
    metrics = GlobalPointerScore()

    max_f1 = 0.
    for epoch in range(configs.num_train_epoch):
        train_sampler.set_epoch(epoch)
        print("Rank:{} - Epoch {}/{}\n".format(local_rank, epoch, configs.num_train_epoch - 1))

        avg_loss = train_ddp(model, train_dataloader, optimizer, scheduler, device, fgm, scaler)
        if local_rank == 0:
            valid_f1, _, _ = valid_ddp(model, valid_dataloader, metrics, device)
            output_writer.add_scalar("loss", avg_loss, epoch)
            output_writer.add_scalar("f1", valid_f1, epoch)

            if valid_f1 > max_f1:
                max_f1 = valid_f1
                if max_f1 > configs.f1_save_threshold:
                    model_f1_val = int(round(max_f1, 3) * 1000)
                    torch.save(model.module.state_dict(),
                               os.path.join(configs.model_save_path, "gp_{}.pt".format(model_f1_val)))

            print(f"Best F1: {max_f1}")

        print(f"Rank:{local_rank} waiting before the barrier")
        dist.barrier()
        print(f"Rank:{local_rank} left the barrier")


if __name__ == "__main__":
    if configs.is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        main_ddp()
    else:
        main()
