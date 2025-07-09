from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/ydtan/XTrack/got10k_lmdb'
    settings.got10k_path = '/home/ydtan/XTrack/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/ydtan/XTrack/itb'
    settings.lasot_extension_subset_path_path = '/home/ydtan/XTrack/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/ydtan/XTrack/lasot_lmdb'
    settings.lasot_path = '/home/ydtan/XTrack/lasot'
    settings.network_path = '/home/ydtan/XTrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/ydtan/XTrack/nfs'
    settings.otb_path = '/home/ydtan/XTrack/otb'
    settings.prj_dir = '/home/zwu-guest123/XTrack'
    settings.result_plot_path = '/home/ydtan/XTrack/output/test/result_plots'
    settings.results_path = '/home/ydtan/XTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/ydtan/XTrack/output'
    settings.segmentation_path = '/home/ydtan/XTrack/output/test/segmentation_results'
    settings.tc128_path = '/home/ydtan/XTrack/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/ydtan/XTrack/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/ydtan/XTrack/trackingnet'
    settings.uav_path = '/home/ydtan/XTrack/uav'
    settings.vot18_path = '/home/ydtan/XTrack/vot2018'
    settings.vot22_path = '/home/ydtan/XTrack/vot2022'
    settings.vot_path = '/home/ydtan/XTrack/VOT2019'
    settings.youtubevos_dir = ''

    return settings

