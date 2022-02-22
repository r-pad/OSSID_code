from .dtoid import DtoidNet

def getModel(cfg):
    if cfg.model.name == "dtoid":
        ModelClass = DtoidNet
    else:
        raise Exception("Unknown cfg.model.name =", cfg.model.name)

    if cfg.weights_path is None:
        model = ModelClass(cfg)
    else:
        print("Loading Model from checkpoint:", cfg.weights_path)
        model = ModelClass.load_from_checkpoint(cfg.weights_path, config=cfg)
    
    return model