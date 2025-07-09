# XTrack
Official implement of XTrack(XTrack: Multimodal Training Boosts RGB-X Video Object Trackers).

:star: :star: :star: ICCV 2025 :star: :star: :star:

Paper: [[Preprint](https://arxiv.org/pdf/2405.17773)].

Data: [🤗 VOT-RGBD2022](https://huggingface.co/datasets/taryya/VOT-RGBD202) 

Model Weight: [🤗XTrack-Base&OSTrack&DropMAE](https://huggingface.co/taryya/XTrack) 

Raw Result:[Google Drive](https://drive.google.com/drive/folders/1GamVMv4v7OcYeu_xFynck6Odb-9-QtKq?usp=drive_link)

### Mixture of Modal Experts
![meme_pipeline](https://github.com/supertyd/XTrack/blob/main/meme_pipeline.gif)





### SOTA Comparison
| Method       | DepthTrack (F-score↑) | DepthTrack (Re↑) | DepthTrack (Pr↑) | VOT-RGBD2022 (EAO↑) | VOT-RGBD2022 (Acc.↑) | VOT-RGBD2022 (Rob.↑) | LasHeR (Pr↑) | LasHeR (Sr↑) | RGBT234 (MPR↑) | RGBT234 (MSR↑) | VisEvent (Pr↑) | VisEvent (Sr↑) |
|--------------|-----------------------|------------------|------------------|---------------------|----------------------|----------------------|--------------|-------------|----------------|----------------|----------------|----------------|
| **XTrack-L** | **64.8**              | **64.3**         | **65.4**         | **74.0**            | **82.8**             | **88.9**             | **73.1**     | **58.7**    | **87.8**       | **65.4**       | **80.5**       | **63.3**       |
| **XTrack-B** | 61.8                  | **62.0**         | **61.5**         | **74.0**            | **82.1**             | **88.8**             | **69.1**     | **55.7**    | **87.4**       | **64.9**       | **77.5**       | **60.9**       |
| **UnTrack**  | 61.0                  | 61.0             | 61.0             | 72.1                | 82.0                 | 86.9                 | 64.6         | 51.3        | 84.2           | 62.5           | 75.5           | 58.9           |
| **SDSTrack** | **61.9**              | 60.9             | 61.4             | 72.8                | 81.2                 | 88.3                 | 66.5         | 53.1        | 84.8           | 62.5           | 76.7           | 59.7           |
| **OneTracker** | 60.9                | 60.4             | 60.7             | 72.7                | 81.9                 | 87.2                 | 67.2         | 53.8        | 85.7           | 64.2           | 76.7           | 60.8           |
| **ViPT**     | 59.4                  | 59.6             | 59.2             | 72.1                | 81.5                 | 87.1                 | 65.1         | 52.5        | 83.5           | 61.7           | 75.8           | 59.2           |
| **ProTrack** | 57.8                  | 57.3             | 58.3             | 65.1                | 80.1                 | 80.2                 | 53.8         | 42.0        | 79.5           | 59.9           | 63.2           | 47.1           |





## Usage
### Installation
Create and activate a conda environment:
```
conda create -n xtrack python=3.7
conda activate xtrack
```
Install the required packages:
```
bash install_xtrack.sh
```

### Data Preparation
Download the training datasets, It should look like:
```
$<PATH_of_Datasets>
    -- LasHeR/TrainingSet
        |-- 1boygo
        |-- 1handsth
        ...
    -- VisEvent/train
        |-- 00142_tank_outdoor2
        |-- 00143_tank_outdoor2
        ...
        |-- trainlist.txt
```

### Path Setting
Run the following command to set paths:
```
cd <PATH_of_XTrack>
python tracking/create_default_local_file.py --workspace_dir . --data_dir <PATH_of_Datasets> --save_dir ./output
```
You can also modify paths by these two files:
```
./lib/train/admin/local.py  # paths for training
./lib/test/evaluation/local.py  # paths for testing
```

### Training
Dowmload the pretrained foundation model as posted above.
and put it under ./pretrained/.
```
bash train_xtrack.sh
```
You can train models with various modalities and variants by modifying ```train_xtrack.sh```.

### Testing

#### For RGB-T benchmarks
[LasHeR & RGBT234] \
Modify the <DATASET_PATH> and <SAVE_PATH> in```./RGBT_workspace/test_rgbt_mgpus.py```, then run:
```
bash eval_rgbt.sh
```
We refer you to use [LasHeR Toolkit](https://github.com/BUGPLEASEOUT/LasHeR) for LasHeR evaluation, 
and refer you to use [MPR_MSR_Evaluation](https://sites.google.com/view/ahutracking001/) for RGBT234 evaluation.


#### For RGB-E benchmark
[VisEvent]\
Modify the <DATASET_PATH> and <SAVE_PATH> in```./RGBE_workspace/test_rgbe_mgpus.py```, then run:
```
bash eval_rgbe.sh
```
We refer you to use [VisEvent_SOT_Benchmark](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark) for evaluation.

## Citation
Please cite our work if you think it is useful for your research.

```bibtex

@inproceedings{Tan2024XTrack,
  author    = {Yuedong Tan and Zongwei Wu and Yuqian Fu and Zhuyun Zhou and Guang Sun and Chang-Bin Ma and Danda Pani Paudel and Luc Van Gool and Radu Timofte},
  title     = {XTrack: Multimodal Training Boosts RGB-X Video Object Trackers},
  booktitle   = {ICCV 2025},
  year      = {2025},

}
```





## Acknowledgment
- This repo is based on [BAT](https://github.com/SparkTempest/BAT) 
- Thanks for the [OSTrack](https://github.com/botaoye/OSTrack) and [PyTracking](https://github.com/visionml/pytracking) library.
