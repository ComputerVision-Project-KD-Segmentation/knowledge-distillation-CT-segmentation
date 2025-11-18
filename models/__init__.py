from .ENet import ENet
from .RAUNet import RAUNet

def get_model(model_name: str, channels: int):
    assert model_name.lower() in ['enet', 'raunet']
    if model_name.lower() == 'raunet':
        model = RAUNet(num_classes=channels)
    elif model_name.lower() == 'enet':
        model = ENet(num_classes=channels)
    return model