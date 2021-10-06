
from ML_training import Train

#trainer = Train.Trainer(total_len=total_len, dim=[[0, 0, 1],[1, 1, 1, 1, 1, 1]], model_type="Actor_Critic", batch_size = 32)
trainer = Train.Trainer(dim=[[0, 0, 1],[1, 1, 1, 1, 1, 1]], model_type="DQN_2", batch_size = 32, datasets_directory="C:/ML_car/Datasets/Preprocessed/fsebcardom_996df78",
                        load_model=False)

#trainer.simulate_on_datasets(visualise=True, type=1)
trainer.simulate_on_dataset(file_name="f_s_e_b_c_a_r_d_o_m_combined.npy")
