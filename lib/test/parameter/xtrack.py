from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.xtrack.config import cfg, update_config_from_file


def parameters(yaml_name: str, epoch=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/xtrack/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    params.checkpoint = os.path.join(prj_dir, "output/checkpoints/train/xtrack/rgbt/XTrack_ep%04d.pth.tar" % (cfg.TEST.EPOCH))
    # params.checkpoint = os.path.join(prj_dir, "models/BAT_%s.pth"%yaml_name)
    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
