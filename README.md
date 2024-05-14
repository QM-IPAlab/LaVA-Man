# TODO list
## task: the netowork structure (ssl) 
9-May, Thursday
- [x] 像croco一样，需要两张图片+text 作为一个pair
- [x] 用stable diffusion的方式跑mae
- [x] Cross attention 原图不挖空，结果图挖空。原图是Q，还是K V？

## task: use pretrain model in simulator tasks
12-May, Sunday
- [ ] write the model in cliport agent
- [ ] run the experiment

### 其他想法
- [ ] 不同的视角？？ （contribution）
- [ ] decoder 要不要丢掉？
- [ ] 更换文字模版 （why:更伟大的贡献）
- [ ] encoder里也加上text cross attention

### Deadlines
NIPS:
Abstract submission: 15 May \
Paper submission: 22 May

CORL:
Paper submission: 6th

RAL: JULY

ICRA: Sep
ICLR: Sep


### Baselines
- CLIPort from https://github.com/cliport/cliport
- MAE from origial: https://github.com/facebookresearch/mae


### Install
Install MAE environment here : https://github.com/facebookresearch/deit/blob/main/README_deit.md

Install PyTorch==1.7.1 and timm==0.3.2
`conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch`

