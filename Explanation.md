# Components
1. system: a LightningTorchModule (i.e. a trainable model), that contains the gaussians, and implements training_step (i.e. forward() method). Defined in 
gaussian_object_system.py. 
2. dm: a LightningTorchDataModule, handles data. There are both 'train' LooDataset and 'test' LooDataset. Defined in loo.py. For every __getitem__ call, returns:
  a. train camera + its image
  b. random_poses - a list[] of tuples (R,T), where R is (3,3) and T is (3,). 
  c. gt_images (the training images)
  d. random_R, random_T =  random_poses[random_idx], where random_idx was chosen randomly between 0 and len(random_poses)
4. Trainer - combines system and dm. 

# Training Step
1. Compute max_cam_dis - maximum distance between camera positions
2. 
