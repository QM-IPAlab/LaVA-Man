from cliport.agents.transporter_lang_goal import TwoStreamClipLingUNetTransporterAgent
from cliport.models.streams.two_stream_attention_lang_fusion import TwoStreamAttentionMAEFixSize
from cliport.models.streams.two_stream_transport_lang_fusion import TwoStreamTransportMAEFixSize
from cliport.utils import utils

import numpy as np
import torch


class MAESeg2TwoStreamTransporterAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet_lat'
        stream_two_fcn = 'mae_seg2_lat'
        self.attention = TwoStreamAttentionMAEFixSize(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportMAEFixSize(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )