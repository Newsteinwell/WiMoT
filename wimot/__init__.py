"""WiMoT: replay-free continual Wi-Fi sensing via moment transport."""


def __getattr__(name):  # lazy, so that importing the package does not import torch
    if name == 'WiMoT':
        from .api import WiMoT
        return WiMoT
    raise AttributeError(name)
