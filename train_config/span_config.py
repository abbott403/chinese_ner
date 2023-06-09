ent2id = {"address": 0, "book": 1, "company": 2, "game": 3, "government": 4, "movie": 5, "name": 6, "organization": 7,
          "position": 8, "scene": 9}

train_data_path = "data/"
pretrained_model_path = "third_party_weights/bert_base_chinese/"
model_save_path = "./model_weights"
scheduler = "CAWR"

cawr_scheduler = {"T_mult": 1, "rewarm_epoch_num": 2}
step_scheduler = {"decay_rate": 0.999, "decay_steps": 200}
f1_save_threshold = 0.79
batch_size = 64
num_train_epoch = 50
max_len = 128
seed = 42
num_work_load = 0
learning_rate = 5e-5
