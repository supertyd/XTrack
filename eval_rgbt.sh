#NCCL_P2P_LEVEL=NVL python tracking/train.py --script bat --config rgbt --save_dir ./output --mode single


cd Depthtrack_workspace
vot evaluate --workspace ./ xtrack
vot analysis --nocache --name xtrack
cd ..




CUDA_VISIBLE_DEVICES=0 NCCL_P2P_LEVEL=NVL python ./RGBE_workspace/test_rgbe_mgpus.py --script_name xtrack --dataset_name VisEvent --yaml_name rgbe
# test lasher
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_LEVEL=NVL python ./RGBT_workspace/test_rgbt_mgpus.py --script_name xtrack --dataset_name LasHeR --yaml_name rgbt

# test rgbt234
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_LEVEL=NVL python ./RGBT_workspace/test_rgbt_mgpus.py --script_name xtrack --dataset_name RGBT234 --yaml_name rgbt



#CUDA_VISIBLE_DEVICES=0 NCCL_P2P_LEVEL=NVL python ./RGBT_workspace/test_rgbt_mgpus.py --script_name xtrack --dataset_name DroneT --yaml_name rgbt


#CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_LEVEL=NVL python ./RGBT_workspace/test_rgbt_mgpus.py --script_name xtrack --dataset_name VTUAVST --yaml_name rgbt






