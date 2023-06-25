entities = ["O", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER"]
ent2id = {ent: idx for idx, ent in enumerate(entities)}
id2ent = {idx: ent for idx, ent in enumerate(entities)}

train_data_path = "data/msra/"
pretrained_model_path = "third_party_weights/bert_base_chinese/"
model_save_path = "./model_weights/bert_softmax_crf/"
scheduler = "Linear"

cawr_scheduler = {"T_mult": 1, "rewarm_epoch_num": 2}
step_scheduler = {"decay_rate": 0.999, "decay_steps": 200}
f1_save_threshold = 0.90
batch_size = 32
num_train_epoch = 50
dropout_rate = 0.1
seed = 42
num_work_load = 2
softmax_learning_rate = 1e-3
crf_learning_rate = 1e-4
is_ddp = True
nprocs_per_node = 4
use_attack = False
use_amp = True
loss_type = "cross_entropy"
