"""
evaluate_refer.py

copy from voltron
Example script for loading a pretrained V-Cond model (from the `voltron` library), configuring a MAP-based extractor
factory function, and then defining/invoking the ReferDetectionHarness.

To run this script, pip install the voltron and voltron_eval first
"""
import torch
import mae.models_lib as models_lib
from voltron import instantiate_extractor, load
from mae.eval_voltron_tools import ReferDetectionHarness
from mae.main_pretrain_ours import get_args_parser
from mae.util import misc

def evaluate_refer(args) -> None:
    # Load Backbone (V-Cond)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone, preprocess = load("v-cond", device=device)
    del backbone

    # Load our model
    model = models_lib.__dict__[args.model](img_size=(224,224))
    if args.pretrain:
        misc.dynamic_load_pretrain(model, args.pretrain, interpolate=True)
    backbone = model

    # Create MAP Extractor Factory (single latent =>> we only predict of a single dense vector representation)
    map_extractor_fn = instantiate_extractor(backbone, n_latents=1)

    # Create Refer Detection Harness
    refer_evaluator = ReferDetectionHarness(args.model, backbone, preprocess, map_extractor_fn, args=args)
    refer_evaluator.fit()
    refer_evaluator.test()


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    evaluate_refer(args)
