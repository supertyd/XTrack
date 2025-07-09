class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/zwu-guest123/XTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/zwu-guest123/XTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/zwu-guest123/ydtan/XTrack/pretrained_networks'
        self.got10k_val_dir = '/home/zwu-guest123/XTrack/got10k/val'
        self.lasot_lmdb_dir = '/home/zwu-guest123/XTrack/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/zwu-guest123/XTrack/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/home/zwu-guest123/XTrack/trackingnet_lmdb'
        self.coco_lmdb_dir = '/home/zwu-guest123/XTrack/coco_lmdb'
        self.coco_dir = '/home/ydtan/XTrack/coco'
        self.lasot_dir = '/home/ydtan/XTrack/lasot'
        self.got10k_dir = '/home/ydtan/XTrack/got10k/train'
        self.trackingnet_dir = '/home/ydtan/XTrack/trackingnet'
        self.depthtrack_dir = '/home/zwu-guest123/data/depthtrack/train'
        self.lasher_dir = '/home/zwu-guest123/data/lasher/trainingset'
        self.visevent_dir = '/home/zwu-guest123/data/visevent/train'


