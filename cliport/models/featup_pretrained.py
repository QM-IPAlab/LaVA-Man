import torch
import torch.nn as nn

class FeatUp(nn.Module):
    def __init__(self, pretrain=False):
        super(FeatUp, self).__init__()
        featup_model = torch.hub.load("/jmain02/home/J2AD007/txk47/cxz00-txk47/.cache/torch/hub/mhamilton723_FeatUp_main", 
                                      'clip', source='local', pretrained=pretrain)
        self.upsampler = featup_model.upsampler

    def forward(self, source, guidance):
        return self.upsampler(source, guidance)

