entities = ["O", "LOC", "ORG", "PER"]
ent2id = {ent: idx for idx, ent in enumerate(entities)}
id2ent = {idx: ent for idx, ent in enumerate(entities)}

train_data_path = "data/msra/"
pretrained_model_path = "third_party_weights/bert_base_chinese/"
model_save_path = "./model_weights"
scheduler = "CAWR"

cawr_scheduler = {"T_mult": 1, "rewarm_epoch_num": 2}
step_scheduler = {"decay_rate": 0.999, "decay_steps": 200}
f1_save_threshold = 0.99
batch_size = 512
num_train_epoch = 500000000
max_len = 512
dropout_rate = 0.1
seed = 42
num_work_load = 1
learning_rate = 1e-4
is_ddp = True
nprocs_per_node = 3
use_attack = True
use_amp = False
loss_type = "cross_entropy"
soft_label = True
