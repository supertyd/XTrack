# Training BAT
#NCCL_P2P_LEVEL=NVL python tracking/train.py --script bat --config rgbt_384 --save_dir ./output --mode single --nproc_per_node 1
#NCCL_P2P_LEVEL=NVL python tracking/train.py --script bat --config rgbt --save_dir ./output --mode single --nproc_per_node 1
python tracking/train.py --script bat --config rgbt --save_dir ./output --mode single --nproc_per_node 1
#python tracking/train.py --script bat --config rgbt --save_dir ./output --mode multiple --nproc_per_node 1 --use_wandb 1
#NCCL_P2P_LEVEL=NVL python tracking/train.py --script bat --config rgbe --save_dir ./output --mode multiple --nproc_per_node 4
#CUDA_VISIBLE_DEVICES=0 NCCL_P2P_LEVEL=NVL python ./RGBT_workspace/test_rgbt_mgpus.py --script_name bat --dataset_name LasHeR --yaml_name rgbt


cd Depthtrack_workspace
vot evaluate --workspace ./ bat
vot analysis --nocache --name bat
cd ..
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_LEVEL=NVL python ./RGBT_workspace/test_rgbt_mgpus.py --script_name bat --dataset_name RGBT234 --yaml_name rgbt
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_LEVEL=NVL python ./RGBT_workspace/test_rgbt_mgpus.py --script_name bat --dataset_name RGBT234 --yaml_name rgbt


# test rgbt234

#
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_LEVEL=NVL python ./RGBE_workspace/test_rgbe_mgpus.py --script_name bat --dataset_name VisEvent --yaml_name rgbe
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_LEVEL=NVL python ./RGBT_workspace/test_rgbt_mgpus.py --script_name bat --dataset_name LasHeR --yaml_name rgbt

CUDA_VISIBLE_DEVICES=0 NCCL_P2P_LEVEL=NVL python ./RGBE_workspace/test_rgbe_mgpus.py --script_name bat --dataset_name VisEvent --yaml_name rgbe
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_LEVEL=NVL python ./RGBT_workspace/test_rgbt_mgpus.py --script_name bat --dataset_name LasHeR --yaml_name rgbt

cd VOT22RGBD_workspace
vot evaluate --workspace ./ bat
vot analysis --nocache --name bat
cd ..



