from lib.models.ddstereo import build_lightstereoOoDdetr, build_lightstereoOoDdetrOnehead

def build_model(cfg):

    if cfg["model_type"] == 'lightstereoOoDdetr':
        return build_lightstereoOoDdetr(cfg)
    elif cfg["model_type"] == 'lightstereoOoDdetrOnehead':
        return build_lightstereoOoDdetrOnehead(cfg)
    else:
        raise NotImplementedError("Model type '{}' not recognized.".format(cfg["model_type"]))
