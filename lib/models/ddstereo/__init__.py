from .stereolightdetr_Ood import build_lightstereoOod
from .stereolightdetr_OodOnehead import build_lightstereoOodOnehead

def build_lightstereoOoDdetr(cfg):
    return build_lightstereoOod(cfg)

def build_lightstereoOoDdetrOnehead(cfg):
    return build_lightstereoOodOnehead(cfg)
