didi_data_path = '/data/fan/didi/'
didi_traj_dict_path = didi_data_path + 'processed/traj_dict/'
didi_interp_traj_path = didi_data_path + 'processed/interp_traj/'
didi_hash_traj_path = didi_data_path + 'processed/hash_traj/'
didi_hidx_traj_path = didi_data_path + 'processed/hidx_traj/'
hash_dict_path = didi_data_path + 'processed/hash_dict.pk'

ours_predictor_model_path = './crowd_context_gru.pytorch'
context_mean_predictor_model_path = './crowd_context_mean.pytorch'
context_max_predictor_model_path = './crowd_context_max.pytorch'
gru_predictor_model_path = './cluster_gru.pytorch'
conditional_predictor_model_path = './conditional.pytorch'
ensemble_components_folder_path = './ensemble/'

num_clusters = 700
num_timeslots = 240
time_embedding = 128
cluster_embedding = 256
hidden_dim = 256
context_dim = 64
dT = 12

ensemble_cluster_embedding = 16