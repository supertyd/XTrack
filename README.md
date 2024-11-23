# XTrack
Official implement of XTrack, a generalist and blind RGB-X tracker. [[Preprint](https://arxiv.org/pdf/2405.17773)].

### Mixture of Modal Experts
![meme_pipeline](https://github.com/user-attachments/assets/c5ab5341-c2f3-493c-9ea0-3b6457368301)



### Our Results on RGB-Depth,RGB-Event, and RGB-Thermal Datasets
[Google Drive](https://drive.google.com/drive/folders/1GamVMv4v7OcYeu_xFynck6Odb-9-QtKq?usp=drive_link)


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



