from cliport.agents.transporter import OriginalTransporterAgent
from cliport.agents.transporter import ClipUNetTransporterAgent
from cliport.agents.transporter import TwoStreamClipWithoutSkipsTransporterAgent
from cliport.agents.transporter import TwoStreamRN50BertUNetTransporterAgent
from cliport.agents.transporter import TwoStreamClipUNetTransporterAgent

from cliport.agents.transporter_lang_goal import TwoStreamClipLingUNetTransporterAgent
from cliport.agents.transporter_lang_goal import TwoStreamRN50BertLingUNetTransporterAgent
from cliport.agents.transporter_lang_goal import TwoStreamUntrainedRN50BertLingUNetTransporterAgent
from cliport.agents.transporter_lang_goal import OriginalTransporterLangFusionAgent
from cliport.agents.transporter_lang_goal import ClipLingUNetTransporterAgent
from cliport.agents.transporter_lang_goal import TwoStreamRN50BertLingUNetLatTransporterAgent

from cliport.agents.transporter_image_goal import ImageGoalTransporterAgent

from cliport.agents.transporter import TwoStreamClipUNetLatTransporterAgent
from cliport.agents.transporter_lang_goal import TwoStreamClipLingUNetLatTransporterAgent
from cliport.agents.transporter_lang_goal import TwoStreamClipFilmLingUNetLatTransporterAgent

# Ours
from cliport.agents.mae_transporter import MAETransporterAgent, MAESegTransporterAgent, MAEFixTransporterAgent
from cliport.agents.mae_transporter import MAEFixGloss, MAESeg2TransporterAgent, MAESeg2TransporterAgentRenor, MAESeg2DTransporterAgent
from cliport.agents.mae_transporter import MAESeg2ModelFullMaskAgent, MAESeg3TransporterAgent, MAEFeatUpTransporterAgent, MAESegDPT3TransporterAgent
from cliport.agents.mae_transporter import MAESeg2DepthTransporterAgent, MAESegBaseAgent, MAESegCLIPModel, MAESegDPTTransporterAgent, MAESegDPT2LossTransporterAgent
from cliport.agents.mae_transporter_two_stream import MAESeg2TwoStreamTransporterAgent, MAESeg2PlusTwoStreamTransporterAgent
from cliport.agents.sep_transporter import PickAgent, PlaceAgent

from cliport.agents.transporter_sep_models import MAESepSeg2Agent, MAESepDPTAgent, MAESepDPTSKAgent, MAESepDPTSegAgent, MAESepSeg2DAgent
from cliport.agents.transporter_sep_models import MAESepCLIP, MAESepBase, MAESepDual, MAESepFullmasked, MAEFozenEncoder, MAESepRecon, MAESepAdd, MAESepAddClipv, MAESepFuse, MAESepFuseFM

from cliport.agents.susie_transporter import MAESepSeg2AgentSusie

names = {

         ### OURS MAE
         'mae': MAETransporterAgent,
         'mae_seg': MAESegTransporterAgent,
         'mae_fixed': MAEFixTransporterAgent,
         'mae_fixed_gloss': MAEFixGloss,
         'mae_seg2': MAESeg2TransporterAgent,
         'mae_seg2_renor':MAESeg2TransporterAgentRenor,
         'mae_seg2_depth': MAESeg2DepthTransporterAgent,
         'mae_seg_base': MAESegBaseAgent,
         'mae_seg2_fm': MAESeg2ModelFullMaskAgent,
         'mae_seg2_lat': MAESeg2TwoStreamTransporterAgent,
         'mae_seg3': MAESeg3TransporterAgent,
         'mae_featup':MAEFeatUpTransporterAgent,
         'mae_seg2_lat_plus': MAESeg2PlusTwoStreamTransporterAgent,
         'mae_clip': MAESegCLIPModel,
         'mae_seg_dpt': MAESegDPTTransporterAgent,
         'mae_seg_dpt_2loss': MAESegDPT2LossTransporterAgent,
         'mae_seg2_dpt': MAESegDPT3TransporterAgent,
         'mae_seg2d': MAESeg2DTransporterAgent,

         'mae_sep_seg2': MAESepSeg2Agent,
         'mae_sep_dpt': MAESepDPTAgent,
         'mae_sep_dpt_sk': MAESepDPTSKAgent,
         'mae_sep_dpt_seg': MAESepDPTSegAgent,
         'mae_sep_seg2d': MAESepSeg2DAgent,
         'mae_sep_clip': MAESepCLIP,
         'mae_sep_base': MAESepBase,
         'mae_seg_recond': MAESepRecon,
         'mae_sep_seg2_add':MAESepAdd,
         'mae_sep_seg2_add_clipv':MAESepAddClipv,
         'mae_fuse':MAESepFuse,
         'mae_fuse_fm':MAESepFuseFM,

         'susie': MAESepSeg2AgentSusie,

         #ablation study:
         'mae_sep_seg2_dual': MAESepDual,
         'mae_sep_seg2_fm': MAESepFullmasked,
         'mae_sep_seg2_froz_e': MAEFozenEncoder,

         ## Separated Transporter
         'pick': PickAgent,
         'place': PlaceAgent,

         ################################
         ### CLIPort ###
         'cliport': TwoStreamClipLingUNetLatTransporterAgent,
         'two_stream_clip_lingunet_lat_transporter': TwoStreamClipLingUNetLatTransporterAgent,

         ################################
         ### Two-Stream Architectures ###
         # CLIPort without language
         'two_stream_clip_unet_lat_transporter': TwoStreamClipUNetLatTransporterAgent,

         # CLIPort without lateral connections
         'two_stream_clip_lingunet_transporter': TwoStreamClipLingUNetTransporterAgent,

         # CLIPort without language and lateral connections
         'two_stream_clip_unet_transporter': TwoStreamClipUNetTransporterAgent,

         # CLIPort without language, lateral, or skip connections
         'two_stream_clip_woskip_transporter': TwoStreamClipWithoutSkipsTransporterAgent,

         # RN50-BERT
         'rn50_bert': TwoStreamRN50BertLingUNetLatTransporterAgent,

         # RN50-BERT without language
         'two_stream_full_rn50_bert_unet_transporter': TwoStreamRN50BertUNetTransporterAgent,

         # RN50-BERT without lateral connections
         'two_stream_full_rn50_bert_lingunet_transporter': TwoStreamRN50BertLingUNetTransporterAgent,

         # Untrained RN50-BERT (similar to untrained CLIP)
         'two_stream_full_untrained_rn50_bert_lingunet_transporter': TwoStreamUntrainedRN50BertLingUNetTransporterAgent,

         ###################################
         ### Single-Stream Architectures ###
         # Transporter-only
         'transporter': OriginalTransporterAgent,

         # CLIP-only without language
         'clip_unet_transporter': ClipUNetTransporterAgent,

         # CLIP-only
         'clip_lingunet_transporter': ClipLingUNetTransporterAgent,

         # Transporter with language (at bottleneck)
         'transporter_lang': OriginalTransporterLangFusionAgent,

         # Image-Goal Transporter
         'image_goal_transporter': ImageGoalTransporterAgent,

         ##############################################
         ### New variants NOT reported in the paper ###

         # CLIPort with FiLM language fusion
         'two_stream_clip_film_lingunet_lat_transporter': TwoStreamClipFilmLingUNetLatTransporterAgent,
         }