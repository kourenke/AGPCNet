import torch
from mmengine import Config

from irstdet.models import AGPCNet
from irstdet.registry import MODELS
from irstdet.utils import get_model_complexity_info,compute_speed

if __name__ == '__main__':

    in_channels = 3
    img_h = 256
    img_w = 256
    input_size = (1, in_channels, img_h, img_w)
    MODEL = dict(type=AGPCNet,
                 backbone='resnet18', 
                 scales=(10, 6), 
                 reduce_ratios=(8, 8), 
                 gca_type='patch', 
                 gca_att='origin',
                 drop=0.1,
                 loss_mask_cfg=dict(type='SoftIoULoss', use_sigmoid=True))

    cfg = Config(MODEL)
    model = MODELS.build(cfg)
    model.eval()

    # test loss and predict
    batch = dict(image=torch.rand(*input_size),
                 gt_mask=torch.randint(0, 2, input_size).float())
    loss_and_predict = model.loss_and_predict(batch)


    # test 
    inputs = torch.rand(*input_size)
    get_model_complexity_info(model, inputs)

    # fps 

    compute_speed(model, input_size, device="cuda:0", iteration=100)