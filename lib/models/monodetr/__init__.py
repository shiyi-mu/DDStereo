from .monodetr import build
from .stereodetr import build_stereo
from .stereolightdetr import build_lightstereo
from .stereolightdetr_Ood import build_lightstereoOod
from .stereolightdetr_OodOnehead import build_lightstereoOodOnehead
from .stereodetr_dh import build_stereo_dh
from .monolightdetr import build_lightmono
def build_monodetr(cfg):
    return build(cfg)

def build_stereodetr(cfg):
    return build_stereo(cfg)

def build_lightstereodetr(cfg):
    return build_lightstereo(cfg)

def build_lightstereoOoDdetr(cfg):
    return build_lightstereoOod(cfg)

def build_lightstereoOoDdetrOnehead(cfg):
    return build_lightstereoOodOnehead(cfg)

def build_lightmonodetr(cfg):
    return build_lightmono(cfg)

def build_stereodetr_dh(cfg):
    return build_stereo_dh(cfg)
