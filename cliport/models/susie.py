import torch
import torch.nn.functional as F

from cliport.models.mae_robot_lang import MAESeg2ModelAdd
from susie.model import create_sample_fn
from susie.jax_utils import initialize_compilation_cache


class MAESusieSeg2ModelAdd(MAESeg2ModelAdd):

    def __init__(self, input_shape, output_dim, cfg, 
                 device, preprocess, model_name='mae_robot_lang', 
                 pretrain_path = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big/checkpoint-160.pth'):
        super(MAESusieSeg2ModelAdd, self).__init__(input_shape, output_dim, cfg, 
                 device, preprocess, model_name,
                 pretrain_path)

        print('Loading Susie model, deleting old models...')
        del self.model
        del self.text_processor

        initialize_compilation_cache()
        self.sample_fn = create_sample_fn("kvablack/susie")
    
    
    def forward(self, x, lang):
        import pdb; pdb.set_trace()
        image_out = self.sample_fn(x, "open the drawer")
        
        x = self.preprocess(x, dist='clip')

        in_type = x.dtype
        in_shape = x.shape
        device = x.device
        rgb = x[:, :3]  # select RGB
        latent1, mask1, ids_restore1 = self.forward_encoder(rgb, mask_ratio=0)
        latent2, mask2, ids_restore2 = self.forward_encoder(rgb, mask_ratio=1.0)
        
        lang_emb = self.get_lang_embed(lang,device)[0]
        if latent1.shape[0] != lang_emb.shape[0]:
            lang_emb = lang_emb.repeat([int(latent1.shape[0]//lang_emb.shape[0]), 1, 1])
        for fuse_block in self.model.fuse_blocks:
            latent1, lang_emb = fuse_block(latent1, lang_emb, attention_mask_v=None, attention_mask_l=None)

        fea1 = self.model.decoder_embed(latent1)
        fea2 = self.model.decoder_embed(latent2)
        
        masked_tokens = self.model.mask_token.repeat(fea2.shape[0],
                                               ids_restore2.shape[1] + 1 - fea2.shape[1], 1)
        fea2_ = torch.cat([fea2[:, 1:, :], masked_tokens], dim=1)  # no cls token
        fea2_ = torch.gather(fea2_, dim=1,
                             index=ids_restore2.unsqueeze(-1).repeat(1, 1, fea2.shape[2]))  # unshuffle
        fea2 = torch.cat([fea2[:, :1, :], fea2_], dim=1)  # append cls token

        decoder_pos_embed = self.model.interpolate_pos_encoding(fea1, self.model.decoder_pos_embed, rgb.shape[2], rgb.shape[3])
        fea1 = fea1 + decoder_pos_embed
        fea2 = fea2 + decoder_pos_embed

        out1 = fea1
        out2 = fea2

        for blk in self.model.decoder_blocks:
            out1, out2 = blk(out1, out2, None)
        out = self.model.decoder_norm(out1)
        recon = self.model.decoder_pred(out)
        recon = recon[:, 1:, :]  # 1, 200, 768
        recon = self.unpatchify_img(recon, rgb.shape[2], rgb.shape[3])

        out = out[:, 1:, :]  # 1, 400, 512
        out = self.unpatchify_feature(out, rgb.shape[2], rgb.shape[3])

        out = self.layer1(out)
        out = self.cat1(out, rgb, recon)
        out = self.layer2(out)
        out = self.cat2(out, rgb, recon)
        out = self.layer3(out)
        out = self.cat3(out, rgb, recon)
        out = self.layer4(out)
        out = self.cat4(out, rgb, recon)

        # incase of different size (patch size = 8)
        if out.shape[-2:] != in_shape[-2:]:
            out = F.interpolate(out, size=(in_shape[-2], in_shape[-1]), mode='bilinear')

        predict = self.conv(out)
        return predict