
def postPrcoessConf(config):
    if config.dataset.name == "ycbv_sift":
        if config.dataset.n_kpts_model is None:
            config.dataset.n_kpts_model = config.dataset.n_kpts
        if config.dataset.n_kpts_obs is None:
            config.dataset.n_kpts_obs = config.dataset.n_kpts
    return config
