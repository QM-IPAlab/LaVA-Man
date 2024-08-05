from cliport.agents.transporter_sep import TransporterAgentSep
from cliport.models.streams.one_stream_attention_lang_fusion import OneStreamAttentionMAEFixSize
from cliport.models.streams.one_stream_transport_lang_fusion import OneStreamTransportMAEFixSize
from cliport.utils import utils


class MAESepSeg2Agent(TransporterAgentSep):
    def __init__(self, name, cfg, train_ds, test_ds, mode):
        super().__init__(name, cfg, train_ds, test_ds, mode)

    def _build_model(self):
        stream_fcn = 'mae_seg2'
        self.attention = OneStreamAttentionMAEFixSize(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportMAEFixSize(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )