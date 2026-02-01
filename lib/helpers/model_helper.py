from lib.models.monodetr import build_monodetr, build_stereodetr, build_lightstereodetr, build_lightstereoOoDdetr, build_stereodetr_dh, build_lightmonodetr, build_lightstereoOoDdetrOnehead

def build_model(cfg):

    if cfg["model_type"] == 'monodetr':
        return build_monodetr(cfg)
    elif cfg["model_type"] == 'stereodetr':
        return build_stereodetr(cfg)
    elif cfg["model_type"] == 'lightstereodetr':
        return build_lightstereodetr(cfg)
    elif cfg["model_type"] == 'lightstereoOoDdetr':
        return build_lightstereoOoDdetr(cfg)
    elif cfg["model_type"] == 'lightstereoOoDdetrOnehead':
        return build_lightstereoOoDdetrOnehead(cfg)
    elif cfg["model_type"] == 'lightMonodetr':
        return build_lightmonodetr(cfg)
    elif cfg["model_type"] == 'stereodetr_dh':
        return build_stereodetr_dh(cfg)
    else:
        raise NotImplementedError("Model type '{}' not recognized.".format(cfg["model_type"]))
