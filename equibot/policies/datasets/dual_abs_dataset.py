from abstract_dataset import AbstractDataset
import os
import numpy as np
import torch


class DualAbsDataset(AbstractDataset):
    def __init__(self, cfg, split, transform=None):
        super(DualAbsDataset, self).__init__(cfg, split, transform)

    def process_select(self, cfg):
        if cfg.task_name == 'tape':
            self.process_tape_dual(cfg)
        else:
            raise NotImplementedError('Task not implemented!')

    # NOTE: only for transfer tape. To generalize, we have to revise the postprocessing of dataset. 
    def process_tape_dual(self, cfg):
        print('Processing hdf5 dataset...')
        data_list = []
        raw_files = self.raw_file_names

        for file_id in range(len(raw_files)):
            file_name = raw_files[file_id]
            
            if 'hdf5' in  file_name:
                hdf5_path = os.path.join(self.root, 'raw', file_name)
                import h5py
                with h5py.File(hdf5_path, 'r') as f:

                    if cfg.tamp_type == 'precondition':
                        conditional_pc = f['start_grasps']['obj_points'][()]
                    elif cfg.tamp_type == 'effect':
                        conditional_pc = f['end_grasps']['obj_points'][()]
                    tgt_size = cfg.num_points
                    sampled_indices = np.random.choice(conditional_pc.shape[0], tgt_size, replace=False)
                    conditional_pc = conditional_pc[sampled_indices]

                    if cfg.tamp_type == 'precondition':
                        grasp_poses = f['start_grasps']['grasp_poses'][()]
                    elif cfg.tamp_type == 'effect':
                        grasp_poses = f['end_grasps']['grasp_poses'][()]

                    grasp_nums = len(grasp_poses)
                    joint_data = f['demo_joint_vals'][()]
                    stage = 'precondition'
                    for i in range(len(joint_data)):
                        left_jpose = joint_data[i][:6].reshape(1, 1, 6)
                        right_jpose = joint_data[i][7:13].reshape(1, 1, 6)

                        # only include jpose before OR after the action
                        stage = self.which_stage(stage, left_jpose, right_jpose)
                        if stage != cfg.tamp_type:
                            continue


                        grasp_id = np.random.randint(0, grasp_nums-1)
                        pc_tensor = torch.tensor(conditional_pc).unsqueeze(0).to(torch.float32)
                        grasp_tensor = torch.tensor(grasp_poses[grasp_id]).to(torch.float32).reshape(1, 4, 4)
                        data = {'left_jpose': left_jpose, 'right_jpose': right_jpose, \
                                 'right_pc': pc_tensor,   'right_grasp':grasp_tensor, \
                                    'left_pc': None, 'left_grasp':None}
                        data_list.append(data)

        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save((data_list, None), self.processed_file_path)
        print('processed all hdf5 file!')
