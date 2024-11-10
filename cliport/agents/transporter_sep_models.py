from cliport.agents.transporter_sep import TransporterAgentSep, TransporterAgentSepRecon
from cliport.models.streams.one_stream_attention_lang_fusion import OneStreamAttentionMAEBatch, OneStreamAttentionMAEBatchRecon
from cliport.models.streams.one_stream_transport_lang_fusion import OneStreamTransportMAEBatch, OneStreamTransportMAEBatchRecon
from cliport.utils import utils
from functools import partial

class MAESepSeg2Agent(TransporterAgentSep):
    def __init__(self, name, cfg, train_ds, test_ds, sep_mode):
        self.sep_mode = sep_mode
        super().__init__(name, cfg, train_ds, test_ds, sep_mode)
    
    def create_attention(self, stream_fcn):
        return OneStreamAttentionMAEBatch(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type
        )

    def create_transport(self, stream_fcn):
        return OneStreamTransportMAEBatch(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type
        )

    def _build_model(self):
        stream_fcn = 'mae_seg2'
        if self.sep_mode == 'pick':
            self.attention = self.create_attention(stream_fcn)
            self.transport = None
        
        elif self.sep_mode == 'place':
            self.transport = self.create_transport(stream_fcn)
            self.attention = None
        
        elif self.sep_mode == 'both':
            self.attention = self.create_attention(stream_fcn)
            self.transport = self.create_transport(stream_fcn)
        
        else:
            raise ValueError(f"Invalid sep_mode: {self.sep_mode}")
        

class MAESepDPTAgent(MAESepSeg2Agent):
    def _build_model(self):
        stream_fcn = 'mae_seg_dpt'
        if self.sep_mode == 'pick':
            self.attention = self.create_attention(stream_fcn)
            self.transport = None
        
        elif self.sep_mode == 'place':
            self.transport = self.create_transport(stream_fcn)
            self.attention = None
        
        elif self.sep_mode == 'both':
            self.attention = self.create_attention(stream_fcn)
            self.transport = self.create_transport(stream_fcn)
        
        else:
            raise ValueError(f"Invalid sep_mode: {self.sep_mode}")


class MAESepDPTSKAgent(MAESepSeg2Agent):
    def _build_model(self):
        stream_fcn = 'mae_seg_dpt_sk'
        if self.sep_mode == 'pick':
            self.attention = self.create_attention(stream_fcn)
            self.transport = None
        
        elif self.sep_mode == 'place':
            self.transport = self.create_transport(stream_fcn)
            self.attention = None
        
        elif self.sep_mode == 'both':
            self.attention = self.create_attention(stream_fcn)
            self.transport = self.create_transport(stream_fcn)
        
        else:
            raise ValueError(f"Invalid sep_mode: {self.sep_mode}")
        

class MAESepDPTSegAgent(MAESepSeg2Agent):
    def _build_model(self):
        stream_fcn1 = 'mae_seg_dpt'
        stream_fcn2 = 'mae_seg2'
        if self.sep_mode == 'pick':
            self.attention = self.create_attention(stream_fcn1)
            self.transport = None
        
        elif self.sep_mode == 'place':
            self.transport = self.create_transport(stream_fcn2)
            self.attention = None
        
        elif self.sep_mode == 'both':
            self.attention = self.create_attention(stream_fcn1)
            self.transport = self.create_transport(stream_fcn2)
        
        else:
            raise ValueError(f"Invalid sep_mode: {self.sep_mode}")


class MAESepSeg2DAgent(MAESepSeg2Agent):
    def _build_model(self):
        stream_fcn = 'mae_seg2_dual'
        if self.sep_mode == 'pick':
            self.attention = self.create_attention(stream_fcn)
            self.transport = None
        
        elif self.sep_mode == 'place':
            self.transport = self.create_transport(stream_fcn)
            self.attention = None
        
        elif self.sep_mode == 'both':
            self.attention = self.create_attention(stream_fcn)
            self.transport = self.create_transport(stream_fcn)
        
        else:
            raise ValueError(f"Invalid sep_mode: {self.sep_mode}")


class MAESepCLIP(MAESepSeg2Agent):
    def _build_model(self):
        stream_fcn = 'mae_clip'
        if self.sep_mode == 'pick':
            self.attention = self.create_attention(stream_fcn)
            self.transport = None
        
        elif self.sep_mode == 'place':
            self.transport = self.create_transport(stream_fcn)
            self.attention = None
        
        elif self.sep_mode == 'both':
            self.attention = self.create_attention(stream_fcn)
            self.transport = self.create_transport(stream_fcn)
        
        else:
            raise ValueError(f"Invalid sep_mode: {self.sep_mode}")


class MAESepDual(MAESepSeg2Agent):
    """
    Copy the input image and replace it with the original masked image
    """
    
    def _build_model(self):
        stream_fcn = 'mae_seg2_dual'
        if self.sep_mode == 'pick':
            self.attention = self.create_attention(stream_fcn)
            self.transport = None
        
        elif self.sep_mode == 'place':
            self.transport = self.create_transport(stream_fcn)
            self.attention = None
        
        elif self.sep_mode == 'both':
            self.attention = self.create_attention(stream_fcn)
            self.transport = self.create_transport(stream_fcn)
        
        else:
            raise ValueError(f"Invalid sep_mode: {self.sep_mode}")


class MAEFozenEncoder(MAESepSeg2Agent):
    """
    Frozen the encoder and only train the decoder during fine-tuning
    """
    def _build_model(self):
        stream_fcn = 'mae_seg2_froz_e'
        if self.sep_mode == 'pick':
            self.attention = self.create_attention(stream_fcn)
            self.transport = None
        
        elif self.sep_mode == 'place':
            self.transport = self.create_transport(stream_fcn)
            self.attention = None
        
        elif self.sep_mode == 'both':
            self.attention = self.create_attention(stream_fcn)
            self.transport = self.create_transport(stream_fcn)
        
        else:
            raise ValueError(f"Invalid sep_mode: {self.sep_mode}")


class MAESepFullmasked(MAESepSeg2Agent):
    """
    Copy use full maseked tokens as masked input
    """
    def _build_model(self):
        stream_fcn = 'mae_seg2_fm'
        if self.sep_mode == 'pick':
            self.attention = self.create_attention(stream_fcn)
            self.transport = None
        
        elif self.sep_mode == 'place':
            self.transport = self.create_transport(stream_fcn)
            self.attention = None
        
        elif self.sep_mode == 'both':
            self.attention = self.create_attention(stream_fcn)
            self.transport = self.create_transport(stream_fcn)
        
        else:
            raise ValueError(f"Invalid sep_mode: {self.sep_mode}")


class MAESepBase(MAESepSeg2Agent):
    def _build_model(self):
        stream_fcn = 'mae_seg_base'
        if self.sep_mode == 'pick':
            self.attention = self.create_attention(stream_fcn)
            self.transport = None
        
        elif self.sep_mode == 'place':
            self.transport = self.create_transport(stream_fcn)
            self.attention = None
        
        elif self.sep_mode == 'both':
            self.attention = self.create_attention(stream_fcn)
            self.transport = self.create_transport(stream_fcn)
        
        else:
            raise ValueError(f"Invalid sep_mode: {self.sep_mode}")


class MAESepRecon(TransporterAgentSepRecon):
    """
    Add the reconstruction loss to the model
    """
    def __init__(self, name, cfg, train_ds, test_ds, sep_mode):
        self.sep_mode = sep_mode
        super().__init__(name, cfg, train_ds, test_ds, sep_mode)

    def create_attention(self, stream_fcn):
            return OneStreamAttentionMAEBatchRecon(
                stream_fcn=(stream_fcn, None),
                in_shape=self.in_shape,
                n_rotations=1,
                preprocess=utils.preprocess,
                cfg=self.cfg,
                device=self.device_type
            )

    def create_transport(self, stream_fcn):
        return OneStreamTransportMAEBatchRecon(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type
        )

    def _build_model(self):
        stream_fcn = 'mae_seg_recond'
        if self.sep_mode == 'pick':
            self.attention = self.create_attention(stream_fcn)
            self.transport = None
        
        elif self.sep_mode == 'place':
            self.transport = self.create_transport(stream_fcn)
            self.attention = None
        
        elif self.sep_mode == 'both':
            self.attention = self.create_attention(stream_fcn)
            self.transport = self.create_transport(stream_fcn)
        
        else:
            raise ValueError(f"Invalid sep_mode: {self.sep_mode}")


class MAESepAdd(MAESepSeg2Agent):
    """
    Copy use full maseked tokens as masked input
    """
    def _build_model(self):
        stream_fcn = 'mae_seg2_add'
        if self.sep_mode == 'pick':
            self.attention = self.create_attention(stream_fcn)
            self.transport = None
        
        elif self.sep_mode == 'place':
            self.transport = self.create_transport(stream_fcn)
            self.attention = None
        
        elif self.sep_mode == 'both':
            self.attention = self.create_attention(stream_fcn)
            self.transport = self.create_transport(stream_fcn)
        
        else:
            raise ValueError(f"Invalid sep_mode: {self.sep_mode}")


class MAESepAddClipv(MAESepSeg2Agent):
    """
    Copy use full maseked tokens as masked input
    """
    def _build_model(self):
        stream_fcn = 'mae_seg2_add_clipv'
        if self.sep_mode == 'pick':
            self.attention = self.create_attention(stream_fcn)
            self.transport = None
        
        elif self.sep_mode == 'place':
            self.transport = self.create_transport(stream_fcn)
            self.attention = None
        
        elif self.sep_mode == 'both':
            self.attention = self.create_attention(stream_fcn)
            self.transport = self.create_transport(stream_fcn)
        
        else:
            raise ValueError(f"Invalid sep_mode: {self.sep_mode}")


