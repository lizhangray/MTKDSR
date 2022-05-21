import logging
logger = logging.getLogger('base')


def get_SPSR(checkpoint):
    from .SPSR_model import SPSRModel
    model = SPSRModel(checkpoint="./Checkpoints/SPSR/SPSR_x4.pth")
    return model.netG


def get_EdgeSPSR(checkpoint):
    from .SPSR_model import EdgeSPSRModel
    if checkpoint is not None:
        checkpoint = "./Checkpoints/SPSR/" + checkpoint
    model = EdgeSPSRModel(checkpoint)
    return model.netG