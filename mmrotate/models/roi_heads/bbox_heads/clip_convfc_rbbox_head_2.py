# Copyright (c) 2024 ✨Challyfilio✨
# 2024/1/22
# 改检测头 v2.0
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_linear_layer

from ...builder import ROTATED_HEADS
from .rotated_bbox_head import RotatedBBoxHead

from typing import List, Union

from .clip import clip
from .clip.simple_tokenizer import SimpleTokenizer

from loguru import logger

device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model_name = 'RN50'  ### ViT-B/32
# ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
target_dataset = 'dota'  ### dota, fair
_tokenizer = SimpleTokenizer()

DATASET_EMBEDDINGS = {
    'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
    'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
    'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
    'coco': 'datasets/metadata/coco_clip_a+cname.npy',
}


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        logger.success('build clip text encoder')

    def forward(self, prompts, tokenized_prompts, prompt_tokens=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND (77,16,512)
        if prompt_tokens is not None:
            x = self.transformer(x + prompt_tokens)
        else:
            x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x.float()


# 单个prompt
# '''
class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, CSC=False, class_token_position='end'):
        super().__init__()
        n_cls = len(classnames)  # get_class_names('dota')
        n_ctx = 16  # number of context vectors
        # ctx_init = "classification: a photo of a, a type of. I like it."  # initialization words
        ctx_init = ""  # initialization words
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        # clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = cfg.INPUT.SIZE[0]
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # add "Shallow"
        self.prompt_tokens = nn.Parameter(torch.zeros(1, n_cls, 1, dtype=dtype))  ############################

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt.to(device)).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        logger.info(f'Initial context: "{prompt_prefix}"')
        logger.info(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.to(device)).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        # self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        self.class_token_position = class_token_position  # 'middle' or 'end' or 'front'

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts, self.prompt_tokens


# '''

# 三个prompt
'''
class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, CSC=False, class_token_position='end'):
        super().__init__()
        n_cls = len(classnames)  # get_class_names('dota')
        n_ctx = 16  # number of context vectors
        # ctx_init = "classification: a photo of a, a type of. I like it."  # initialization words
        ctx_init = ""  # initialization words
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        task_desc = 'object classification:'

        # transformer add "Shallow"
        self.prompt_tokens_end = nn.Parameter(torch.zeros(1, n_cls, 1, dtype=dtype))  ############################
        self.prompt_tokens_mid = nn.Parameter(torch.zeros(1, n_cls, 1, dtype=dtype))  ############################
        self.prompt_tokens_pre = nn.Parameter(torch.zeros(1, n_cls, 1, dtype=dtype))  ############################

        # -------- textual prompt --------
        n_task = len(task_desc.split(" "))

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(task_desc + ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 + n_task: 1 + n_task + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        logger.info(f'Initial context: "{prompt_prefix}"')
        logger.info(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized #16,512

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.to(device)).type(dtype)  # 37,77,512

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("task_describe", embedding[:, 1:1 + n_task, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_task + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        # self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        self.class_token_position = class_token_position  # 'middle' or 'end' or 'front'

    def forward(self):
        # -------- textual prompt --------
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        task = self.task_describe

        # logger.debug(prefix.shape)
        # logger.debug(task.shape)
        # logger.debug(ctx.shape)
        # logger.debug(suffix.shape)

        # if self.class_token_position == "end":
        prompts_end = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                task,
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )  ##########################################################

        # elif self.class_token_position == "middle":
        half_n_ctx = self.n_ctx // 2
        prompts_mid = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = prefix[i: i + 1, :, :]
            task_i = task[i: i + 1, :, :]
            class_i = suffix[i: i + 1, :name_len, :]
            suffix_i = suffix[i: i + 1, name_len:, :]
            ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
            ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
            prompt = torch.cat(
                [
                    prefix_i,  # (1, 1, dim)
                    task_i,
                    ctx_i_half1,  # (1, n_ctx//2, dim)
                    class_i,  # (1, name_len, dim)
                    ctx_i_half2,  # (1, n_ctx//2, dim)
                    suffix_i,  # (1, *, dim)
                ],
                dim=1,
            )
            prompts_mid.append(prompt)
        prompts_mid = torch.cat(prompts_mid, dim=0)  ######################################

        # elif self.class_token_position == "front":
        prompts_pre = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = prefix[i: i + 1, :, :]
            task_i = task[i: i + 1, :, :]
            class_i = suffix[i: i + 1, :name_len, :]
            suffix_i = suffix[i: i + 1, name_len:, :]
            ctx_i = ctx[i: i + 1, :, :]
            prompt = torch.cat(
                [
                    prefix_i,  # (1, 1, dim)
                    task_i,
                    class_i,  # (1, name_len, dim)
                    ctx_i,  # (1, n_ctx, dim)
                    suffix_i,  # (1, *, dim)
                ],
                dim=1,
            )
            prompts_pre.append(prompt)
        prompts_pre = torch.cat(prompts_pre, dim=0)

        # else:
        #     raise ValueError

        return prompts_end, prompts_mid, prompts_pre, self.prompt_tokens_end, self.prompt_tokens_mid, self.prompt_tokens_pre


'''

# 文本提示 TextEncoder
'''
class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name='ViT-B/32'):
        super().__init__()
        self.tokenizer = SimpleTokenizer()
        clip_model, _ = clip.load(model_name, device=device)
        # clip_model.eval()
        self.clip = clip_model
        logger.success('build clip text encoder')

    @property
    def device(self):
        return self.clip.device

    @property
    def dtype(self):
        return self.clip.dtype

    def tokenize(self,
                 texts: Union[str, List[str]],
                 context_length: int = 77) -> torch.LongTensor:
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder['<|startoftext|>']
        eot_token = self.tokenizer.encoder['<|endoftext|>']
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token]
                      for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                st = torch.randint(len(tokens) - context_length + 1,
                                   (1,))[0].item()
                tokens = tokens[st:st + context_length]
            result[i, :len(tokens)] = torch.tensor(tokens)
        return result

    def forward(self, text):
        text = self.tokenize(text).to(device)
        text_features = self.clip.encode_text(text).float()
        return text_features
'''

'''
def get_text_embeddings(dataset='dota',
                        custom_vocabulary=None,
                        prompt_prefix='a photo of '):
    assert (dataset is None) ^ (custom_vocabulary is None), \
        'Either `dataset` or `custom_vocabulary` should be specified.'
    if dataset:
        if dataset in DATASET_EMBEDDINGS:
            return DATASET_EMBEDDINGS[dataset]
        else:
            custom_vocabulary = get_class_names(dataset)
            custom_vocabulary += ('background',)
            # print(custom_vocabulary)

    text_encoder = CLIPTextEncoder(model_name=clip_model_name)  #
    text_encoder.eval()  #
    texts = [prompt_prefix + x for x in custom_vocabulary]  #
    # print(texts)
    # print_log(f'Computing text embeddings for {len(custom_vocabulary)} classes.')
    # embeddings = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    embeddings = text_encoder(texts).to(device)
    return embeddings
'''


def get_class_names(dataset):
    if dataset == 'coco':
        from mmdet.datasets import CocoDataset
        class_names = CocoDataset.METAINFO['classes']
    elif dataset == 'cityscapes':
        from mmdet.datasets import CityscapesDataset
        class_names = CityscapesDataset.METAINFO['classes']
    elif dataset == 'voc':
        from mmdet.datasets import VOCDataset
        class_names = VOCDataset.METAINFO['classes']
    elif dataset == 'openimages':
        from mmdet.datasets import OpenImagesDataset
        class_names = OpenImagesDataset.METAINFO['classes']
    elif dataset == 'lvis':
        from mmdet.datasets import LVISV1Dataset
        class_names = LVISV1Dataset.METAINFO['classes']
    elif dataset == 'dota':
        from mmrotate.datasets import DOTADataset
        class_names = DOTADataset.CLASSES
    elif dataset == 'fair':
        from mmrotate.datasets import FairDataset
        class_names = FairDataset.CLASSES
    else:
        raise TypeError(f'Invalid type for dataset name: {type(dataset)}')
    return class_names


class PoswiseFeedForwardNet_1(nn.Module):
    def __init__(self, vis_dim):
        super(PoswiseFeedForwardNet_1, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(vis_dim, vis_dim // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(vis_dim // 4, vis_dim, bias=False))
        # self.ln = nn.LayerNorm(d_model)

    def forward(self, inputs):  # inputs: [batch_size, seq_len, d_model]
        # residual = inputs
        output = self.fc(inputs)
        # output = self.ln(output)
        return output  # [batch_size, seq_len, d_model]


class PoswiseFeedForwardNet_2(nn.Module):
    def __init__(self, vis_dim):
        super(PoswiseFeedForwardNet_2, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(vis_dim, vis_dim // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(vis_dim // 4, vis_dim, bias=False))
        # self.ln = nn.LayerNorm(d_model)

    def forward(self, inputs):  # inputs: [batch_size, seq_len, d_model]
        # residual = inputs
        output = self.fc(inputs)
        # output = self.ln(output)
        return output  # [batch_size, seq_len, d_model]


# double fc
class FCBD(nn.Module):
    def __init__(self, vis_dim_1, vis_dim_2):
        super(FCBD, self).__init__()
        self.img_ffn = PoswiseFeedForwardNet_1(vis_dim_1)
        self.tex_ffn = PoswiseFeedForwardNet_2(vis_dim_2)

    def forward(self, img_feat, tex_feat):
        self_img_feat = self.img_ffn(img_feat)
        self_tex_feat = self.tex_ffn(tex_feat)
        img_feat = img_feat + self_img_feat
        tex_feat = tex_feat + self_tex_feat
        return img_feat, tex_feat


# Shared_FC
@ROTATED_HEADS.register_module()
class ClipRotatedConvFCBBoxHead2(RotatedBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg

    Args:
        num_shared_convs (int, optional): number of ``shared_convs``.
        num_shared_fcs (int, optional): number of ``shared_fcs``.
        num_cls_convs (int, optional): number of ``cls_convs``.
        num_cls_fcs (int, optional): number of ``cls_fcs``.
        num_reg_convs (int, optional): number of ``reg_convs``.
        num_reg_fcs (int, optional): number of ``reg_fcs``.
        conv_out_channels (int, optional): output channels of convolution.
        fc_out_channels (int, optional): output channels of fc.
        conv_cfg (dict, optional): Config of convolution.
        norm_cfg (dict, optional): Config of normalization.
        init_cfg (dict, optional): Config of initialization.
    """

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(ClipRotatedConvFCBBoxHead2, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.prompt_prefix = 'classification: a photo of a type of i like it'  ###
        # self.custom_vocabulary = get_class_names('dota')  ###
        # self.custom_vocabulary += ('background',)  ###
        self.custom_vocabulary = ('background',).__add__(get_class_names(target_dataset))

        clip_model, preprocess = clip.load(clip_model_name, device=device)
        # clip_model.eval()
        self.logit_scale = clip_model.logit_scale
        logger.success('build logit scale: {}'.format(clip_model.logit_scale))
        self.dtype = clip_model.dtype

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = (5 if self.reg_class_agnostic else 5 *
                                                             self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        if init_cfg is None:
            self.init_cfg += [
                dict(
                    type='Xavier',
                    layer='Linear',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

        # self.text_encoder = CLIPTextEncoder(model_name=clip_model_name)
        # self.texts = [self.prompt_prefix + x for x in self.custom_vocabulary]
        self.prompt_learner = PromptLearner(classnames=self.custom_vocabulary, clip_model=clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model=clip_model)

        self.adapter = FCBD(self.fc_out_channels, self.fc_out_channels)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        """Forward function."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # separate branches
        # 这里之后执行回归，分类用clip
        image_features = x
        x_cls = x
        x_reg = x
        # logger.info(x_cls.shape)  # (1024,1024)
        # logger.info(x_reg.shape)  # (1024,1024)

        #### 2024.01.26 改
        # text_features = self.text_encoder(self.texts)
        # text_features = get_text_embeddings().float()
        # logger.info(text_features.shape)

        #### 2024.01.28 改
        tokenized_prompts = self.tokenized_prompts
        prompts, _ = self.prompt_learner()
        # prompts_end, prompts_mid, prompts_pre, prompt_tokens_end, prompt_tokens_mid, prompt_tokens_pre = self.prompt_learner()  # 2024.2.3
        text_features = self.text_encoder(prompts, tokenized_prompts)
        # text_features_end = self.text_encoder(prompts_end, tokenized_prompts, prompt_tokens_end)  # 2024.2.3
        # text_features_mid = self.text_encoder(prompts_mid, tokenized_prompts, prompt_tokens_mid)  # 2024.2.3
        # text_features_pre = self.text_encoder(prompts_pre, tokenized_prompts, prompt_tokens_pre)  # 2024.2.3
        # # 文本特征融合
        # text_features = text_features_end + text_features_mid + text_features_pre  # 2024.2.3

        # 适配器细化学习
        image_features, text_features = self.adapter(image_features, text_features)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # print(self.logit_scale)
        logit_scale = self.logit_scale.exp()
        cls_score = logit_scale * image_features @ text_features.t()
        # logger.info(cls_score.shape)

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score_1 = self.fc_cls(x_cls) if self.with_cls else None
        # logger.info(cls_score_1.shape)
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score + cls_score_1, bbox_pred


'''
# Adapter
@ROTATED_HEADS.register_module()
class RotatedConvFCBBoxHead(RotatedBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg

    Args:
        num_shared_convs (int, optional): number of ``shared_convs``.
        num_shared_fcs (int, optional): number of ``shared_fcs``.
        num_cls_convs (int, optional): number of ``cls_convs``.
        num_cls_fcs (int, optional): number of ``cls_fcs``.
        num_reg_convs (int, optional): number of ``reg_convs``.
        num_reg_fcs (int, optional): number of ``reg_fcs``.
        conv_out_channels (int, optional): output channels of convolution.
        fc_out_channels (int, optional): output channels of fc.
        conv_cfg (dict, optional): Config of convolution.
        norm_cfg (dict, optional): Config of normalization.
        init_cfg (dict, optional): Config of initialization.
    """

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(RotatedConvFCBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        # self.shared_convs, self.shared_fcs, last_layer_dim = \
        #     self._add_conv_fc_branch(
        #         self.num_shared_convs, 1, self.in_channels,
        #         True)
        last_layer_dim = self.in_channels * self.roi_feat_area
        self.adapter_in_channels = last_layer_dim

        self.fc = nn.Linear(self.adapter_in_channels, self.fc_out_channels)
        # self.shared_out_channels = last_layer_dim

        # add cls specific branch
        # self.cls_last_dim --> self.shared_out_channels
        # self.cls_convs, self.cls_fcs, self.cls_last_dim = \
        #     self._add_conv_fc_branch(
        #         self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        # self.reg_last_dim --> self.shared_out_channels
        # self.reg_convs, self.reg_fcs, self.reg_last_dim = \
        #     self._add_conv_fc_branch(
        #         self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        # if self.num_shared_fcs == 0 and not self.with_avg_pool:
        #     if self.num_cls_fcs == 0:
        #         self.cls_last_dim *= self.roi_feat_area
        #     if self.num_reg_fcs == 0:
        #         self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)

        self.adapter = FCBD(self.fc_out_channels, self.fc_out_channels)

        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                # self.cls_last_dim --> self.fc_out_channels
                in_features=self.fc_out_channels,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = (5 if self.reg_class_agnostic else 5 *
                                                             self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                # self.reg_last_dim --> self.fc_out_channels
                in_features=self.fc_out_channels,
                out_features=out_dim_reg)

        # if init_cfg is None:
        #     self.init_cfg += [
        #         dict(
        #             type='Xavier',
        #             layer='Linear',
        #             override=[
        #                 dict(name='shared_fcs'),
        #                 dict(name='cls_fcs'),
        #                 dict(name='reg_fcs')
        #             ])
        #     ]

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        """Forward function."""
        # logger.error(self.num_shared_convs)
        # if self.num_shared_convs > 0:
        #     for conv in self.shared_convs:
        #         x = conv(x)
        # logger.debug(x.shape)

        # logger.error(self.num_shared_fcs)
        # if self.num_shared_fcs > 0:
        #     if self.with_avg_pool:
        #         logger.error('avg_pool')
        #         x = self.avg_pool(x)
        #
        #     x = x.flatten(1)
        #     logger.error(x.shape)
        #
        #     for fc in self.shared_fcs:
        #         x = self.relu(fc(x))
        # logger.debug(x.shape)

        # ---------------------------------
        x = x.flatten(1)
        x = self.relu(self.fc(x))
        # ---------------------------------

        # separate branches
        x_cls = x
        x_reg = x
        # logger.debug(x.shape)

        # for conv in self.cls_convs:
        #     x_cls = conv(x_cls)
        # if x_cls.dim() > 2:
        #     if self.with_avg_pool:
        #         x_cls = self.avg_pool(x_cls)
        #     x_cls = x_cls.flatten(1)
        # for fc in self.cls_fcs:
        #     x_cls = self.relu(fc(x_cls))
        #
        # for conv in self.reg_convs:
        #     x_reg = conv(x_reg)
        # if x_reg.dim() > 2:
        #     if self.with_avg_pool:
        #         x_reg = self.avg_pool(x_reg)
        #     x_reg = x_reg.flatten(1)
        # for fc in self.reg_fcs:
        #     x_reg = self.relu(fc(x_reg))

        x_cls, x_reg = self.adapter(x_cls, x_reg)
        # logger.debug(x_cls.shape)  # 1024,1024
        # logger.debug(x_reg.shape)  # 1024,1024

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred
'''


# 改这里
@ROTATED_HEADS.register_module()
class ClipRotatedShared2FCBBoxHead2(ClipRotatedConvFCBBoxHead2):
    """Shared2FC RBBox head."""

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(ClipRotatedShared2FCBBoxHead2, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@ROTATED_HEADS.register_module()
class ClipRotatedKFIoUShared2FCBBoxHead2(ClipRotatedConvFCBBoxHead2):
    """KFIoU RoI head."""

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(ClipRotatedKFIoUShared2FCBBoxHead2, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """Loss function."""
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                bbox_pred_decode = self.bbox_coder.decode(
                    rois[:, 1:], bbox_pred)
                bbox_targets_decode = self.bbox_coder.decode(
                    rois[:, 1:], bbox_targets)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                    pos_bbox_pred_decode = bbox_pred_decode.view(
                        bbox_pred_decode.size(0), 5)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                    labels[pos_inds.type(torch.bool)]]
                    pos_bbox_pred_decode = bbox_pred_decode.view(
                        bbox_pred_decode.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                    labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    pred_decode=pos_bbox_pred_decode,
                    targets_decode=bbox_targets_decode[pos_inds.type(
                        torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses
