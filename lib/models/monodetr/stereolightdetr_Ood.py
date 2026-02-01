"""
MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection
"""
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import math
import copy
import cv2

from utils import box_ops
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                            accuracy, get_world_size, interpolate,
                            is_dist_avail_and_initialized, inverse_sigmoid)
from utils.box_ops import  box_iou, box_cxcylrtb_to_xyxy

from .backbone import build_backbone
from .bifpn import BiFPN
from .matcherstable_fg_only import build_fg_Stablematcher
from .matcher import build_matcher
from .depthaware_transformer import build_depthaware_transformer_stereo
from .anchor_transformer import build_anchor_transformer_stereo
from .detr_transformer import build_detr_transformer_stereo
from .depth_predictor import LightStereoDepthPredictor
from .depth_predictor.ddn_loss import DDNLoss
from .depth_predictor import DisparityLoss
from lib.losses.focal_loss import sigmoid_focal_loss, SigmoidFocalLoss
from .dn_components import prepare_for_dn, dn_post_process, compute_dn_loss
from .det_stereo_transformer import build_det_stereo_transformer
from .det_mono_transformer import build_det_mono_transformer

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class StereoLightDETROoD(nn.Module):
    """ This is the StereoLightDETROoD module that performs monocualr 3D object detection """
    def __init__(self, backbone, stereo_transformer, mono_transformer, depth_predictor, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, with_center_refine=False, init_box=False,
                 group_num=11, depth_pre_type="one", share_Q=True,depth_sample_mode="3Dcenter",fusion_mode=None, num_scale=1, FPN_type=None,
                 Ood_det=None, dim3d_mode="from_mono"):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            stereo_transformer: depth-aware transformer architecture. See stereo_transformer.py
            mono_transformer: depth-aware transformer architecture. See mono_transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For KITTI, we recommend 50 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            with_center_refine: offset to sample depth
        """
        super().__init__()
 
        self.num_queries = num_queries
        self.stereo_transformer = stereo_transformer
        self.mono_transformer = mono_transformer
        self.depth_predictor = depth_predictor
        hidden_dim = stereo_transformer.d_model
        self.with_center_refine = with_center_refine
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.label_enc = nn.Embedding(num_classes + 1, hidden_dim - 1)  # # for indicator
        # prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        num_classes_fg = 1
        self.class_embed_fg = nn.Linear(hidden_dim, num_classes_fg)
        self.Ood_det = Ood_det
        self.depth_pre_type = depth_pre_type
        self.share_Q = share_Q
        self.depth_sample_mode = depth_sample_mode
    
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        self.class_embed_fg.bias.data = torch.ones(num_classes_fg) * bias_value
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
        self.dim_embed_3d = MLP(hidden_dim, hidden_dim, 3, 2)
        self.angle_embed = MLP(hidden_dim, hidden_dim, 24, 2)
        self.depth_embed = MLP(hidden_dim, hidden_dim, 2, 2)  # depth and deviation
        if self.with_center_refine:
            self.center_refine_offset = MLP(hidden_dim, hidden_dim, 2, 2)  # center refine for depth sample
        
        if self.depth_pre_type in [ "sample_cat" ]:
            self.depth_fuse_conv =  MLP(256+256, 256, 256, 2)
            
        self.fusion_mode = fusion_mode
        self.num_scale = num_scale
        self.FPN_type  = FPN_type
        self.dim3d_mode = dim3d_mode
        if self.FPN_type == "BiFPN":
            self.bifpn = BiFPN(size=[256, 256, 256], feature_size=256, num_layers=3)
        if init_box == True:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
            if self.with_center_refine:
                nn.init.constant_(self.center_refine_offset.layers[-1].weight.data, 0)
                nn.init.constant_(self.center_refine_offset.layers[-1].bias.data, 0)

       
      
        self.query_embed = nn.Embedding(num_queries * group_num, hidden_dim*2)
        if not self.share_Q:
            self.query_embed_mono = nn.Embedding(num_queries * group_num, hidden_dim)
       
        print(">>>>>mushiyi>>>>1029 backbone.num_channels", backbone.num_channels)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
            
        input_proj_list_stereo = []
        for in_channels_disp in [24,24,12,12]:
            input_proj_list_stereo.append(nn.Sequential(
                nn.Conv2d(in_channels_disp, hidden_dim, kernel_size=1),
                nn.GroupNorm(4, hidden_dim),
            ))
           
        self.input_proj_list_stereo = nn.ModuleList(input_proj_list_stereo)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        
        self.num_classes = num_classes


        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (mono_transformer.decoder.num_layers + 1)
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.class_embed_fg = _get_clones(self.class_embed_fg, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.mono_transformer.decoder.bbox_embed = self.bbox_embed
            self.stereo_transformer.decoder.bbox_embed = self.bbox_embed
            self.dim_embed_3d = _get_clones(self.dim_embed_3d, num_pred)
            self.stereo_transformer.decoder.dim_embed = self.dim_embed_3d  
            self.angle_embed = _get_clones(self.angle_embed, num_pred)
            self.depth_embed = _get_clones(self.depth_embed, num_pred)
            if self.depth_pre_type in [ "sample_cat" ]:
                self.depth_fuse_conv =  _get_clones(self.depth_fuse_conv, num_pred)
            
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.class_embed_fg = nn.ModuleList([self.class_embed_fg for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.dim_embed_3d = nn.ModuleList([self.dim_embed_3d for _ in range(num_pred)])
            self.angle_embed = nn.ModuleList([self.angle_embed for _ in range(num_pred)])
            self.depth_embed = nn.ModuleList([self.depth_embed for _ in range(num_pred)])
            if self.depth_pre_type in [ "sample_cat" ]:
                self.depth_fuse_conv =  nn.ModuleList([self.depth_fuse_conv for _ in range(num_pred)])
            self.mono_transformer.decoder.bbox_embed = None

        if self.with_center_refine:
            if with_box_refine:
                self.center_refine_offset = _get_clones(self.center_refine_offset, num_pred)
                nn.init.constant_(self.center_refine_offset[0].layers[-1].bias.data[2:], -2.0)
            else:
                nn.init.constant_(self.center_refine_offset.layers[-1].bias.data[2:], -2.0)
                self.center_refine_offset = nn.ModuleList([self.center_refine_offset for _ in range(num_pred)])

        # 二分之一平均池化
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if self.fusion_mode == "concat":
            self.fuse_conv = nn.Sequential(
                nn.Conv2d(768, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),)
            self.fuse_conv1 = nn.Sequential(
                nn.Conv2d(768, 512, kernel_size=3, padding=1),
                # nn.Dropout2d(0.3),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                # nn.Dropout2d(0.3),
                nn.ReLU(inplace=True),)
            self.fuse_conv2 = nn.Sequential(
                nn.Conv2d(768, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),)
        elif self.fusion_mode == "replace":
              self.fuse_conv = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True))
            
        # if self.fusion_mode == "concat_dropout":
        self.dnn_loss_temp = DDNLoss(depth_sort_reverse=False, 
                                     bg_value=60, 
                                     shrink_ratio=1,
                                     align_by_3d_center=True)

    def sample_from_depth_map(self, depth_logits, targets):
        print("debug targets", len(targets))
        batch_size = depth_logits.shape[0]
        batch_size, _, H_depth_map, W_depth_map = depth_logits.shape
        targets_list = []
        for key in targets.keys():
                targets[key] = targets[key].to('cuda')
        mask = targets['mask_2d']
        key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin', 'heading_res', 'boxes_3d', 'disp', "random_flip_flag", "random_switch_flag"]
        for bz in range(batch_size):
            target_dict = {}
            for key, val in targets.items():
                if key in key_list:
                    if key in ['disp', "random_flip_flag", "random_switch_flag"]:
                        target_dict[key] = val[bz]
                    else:
                        target_dict[key] = val[bz][mask[bz]]
            targets_list.append(target_dict)
        
        num_gt_per_img = [len(t['boxes']) for t in targets_list]

        gt_boxes2d = torch.cat([t['boxes'] for t in targets_list], dim=0) * \
            torch.tensor([W_depth_map, H_depth_map, W_depth_map, H_depth_map], device='cuda')
        gt_boxes3d = torch.cat([t['boxes_3d'] for t in targets_list], dim=0)* \
            torch.tensor([W_depth_map, H_depth_map,W_depth_map,W_depth_map,H_depth_map,H_depth_map], device='cuda')
        
        target_3dcenter = torch.cat([t['boxes_3d'][:, 0: 2] for t in targets_list], dim=0)* torch.tensor([W_depth_map, H_depth_map], device='cuda')
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
        gt_center_depth = torch.cat([t['depth'] for t in targets_list], dim=0).squeeze(dim=1)
        # depth_map = self.dnn_loss_temp.build_target_depth_from_3dcenter(
        #     depth_logits= depth_logits, gt_boxes2d=gt_boxes2d, 
        #     gt_center_depth=gt_center_depth, num_gt_per_img=num_gt_per_img)
        depth_map = self.dnn_loss_temp.build_target_depth_map(
            depth_logits= depth_logits, gt_boxes2d=gt_boxes2d, gt_boxes3d=gt_boxes3d, 
            gt_center_depth=gt_center_depth, num_gt_per_img=num_gt_per_img)
        
        return depth_map
    
    def forward(self, images, calibs, targets, img_sizes, img_sizes_ori, img_sizes_upper, dn_args=None):
        """?The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """

        batch_size = images.shape[0]
        left_images = images[:,0:3,:,:]
        right_images = images[:,3:,:,:]
        images = torch.cat([left_images, right_images], dim=0)
        features, pos_stereo = self.backbone(images)

        srcs_stereo = []
        masks_stereo = []

        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            src_left_right = self.input_proj[l](src)
            
            srcs_stereo.append(src_left_right)
            masks_stereo.append(mask)
            
            assert mask is not None

        if self.num_feature_levels > len(srcs_stereo):
            _len_srcs = len(srcs_stereo)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs_stereo[-1])
                m = torch.zeros(src.shape[0], src.shape[2], src.shape[3]).to(torch.bool).to(src.device)
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs_stereo.append(src)
                masks_stereo.append(mask)
                pos_stereo.append(pos_l)
        srcs_left = []
        masks_left = []
        srcs_right = []
        masks_right = []
        pos_left = []
        
        for src_stereo, mask_stereo, pos_stereo_i in zip(srcs_stereo, masks_stereo, pos_stereo):
        
            src_left = src_stereo[:batch_size]
            mask_left = mask_stereo[:batch_size]
            pos_left_i = pos_stereo_i[:batch_size]
            srcs_left.append(src_left)
            masks_left.append(mask_left)
            pos_left.append(pos_left_i)

            src_right = src_stereo[batch_size:]
            mask_right = mask_stereo[batch_size:]
            srcs_right.append(src_right)
            masks_right.append(mask_right)

        if self.training:
            query_embeds = self.query_embed.weight
            if not self.share_Q:
                query_embeds_for_mono = self.query_embed_mono.weight
        else:
            # only use one group in inference
            query_embeds = self.query_embed.weight[:self.num_queries]
            if not self.share_Q:
                query_embeds_for_mono = self.query_embed_mono.weight[:self.num_queries]
    
        pred_depth_map_logits, depth_pos_embed, weighted_depth,\
              depth_pos_embed_ip, features_for_decoder, pred_disp, feature_depth_dead = self.depth_predictor(
            srcs_stereo, masks_left[2], pos_left[2], targets)
      
        for i in range(0, len(features_for_decoder)):
            features_for_decoder[i]  = self.input_proj_list_stereo[i](features_for_decoder[i])

        intermediate_output = self.stereo_transformer(features_for_decoder[1:4], masks_left[1:4], pos_left[1:4], query_embeds)
        
        hs_stereo = intermediate_output['hs']
        init_reference_stereo = intermediate_output['init_reference_out']
        inter_references_stereo = intermediate_output['inter_references_out']
        
        inter_coords = []
        inter_classes = []
        outputs_3d_dims = []
        # 基于视差的预测分支 预测是否为前景
        for lvl in range(hs_stereo.shape[0]):
            # fg classes
            if lvl == 0:
                reference = init_reference_stereo
            else:
                reference = inter_references_stereo[lvl - 1]
            reference = inverse_sigmoid(reference)

            tmp = self.bbox_embed[lvl](hs_stereo[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            # 3d center + 2d box
            inter_coord = tmp.sigmoid()
            inter_coords.append(inter_coord)
            # classes
            inter_class = self.class_embed_fg[lvl](hs_stereo[lvl])
            inter_classes.append(inter_class)
            
            if self.dim3d_mode == "from_stereo":
                size3d = self.dim_embed_3d[lvl](hs_stereo[lvl])
                outputs_3d_dims.append(size3d)
        
        inter_coords = torch.stack(inter_coords)
        inter_class = torch.stack(inter_classes)
        if self.share_Q:
            query_embeds_for_mono = hs_stereo[-1]
        else:
            query_embeds_for_mono = query_embeds_for_mono.repeat(batch_size, 1, 1)

        hs_mono, init_reference_mono, inter_references_mono = self.mono_transformer(srcs_left[1:4], masks_left[1:4], pos_left[1:4], intermediate_output, query_embeds_for_mono, depth_pos_embed)       
        
        outputs_coords = []
        outputs_samples = []
        outputs_classes = []
        
        
        outputs_depths = []
        outputs_angles = []
        # 基于单目输入的分支
        for lvl in range(hs_mono.shape[0]):
            if lvl == 0:
                reference = init_reference_mono
            else:
                reference = inter_references_mono[lvl - 1]
            reference = inverse_sigmoid(reference)

            tmp = self.bbox_embed[lvl](hs_mono[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            # 3d center + 2d box
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord[..., 0:6])

            # classes
            outputs_class = self.class_embed[lvl](hs_mono[lvl])
            outputs_classes.append(outputs_class)
            
            sample_ponit = outputs_coord[..., 0:2].detach()
            outputs_samples.append(sample_ponit)
            outputs_samples_grid =  ((sample_ponit[..., :2] - 0.5) * 2).unsqueeze(2)
            
            depth_map = F.grid_sample(
                weighted_depth.unsqueeze(1),
                outputs_samples_grid,
                mode='bilinear',
                align_corners=True).squeeze(1)
        
            feature_for_depth_reg = hs_mono[lvl]
            
            depth_reg = self.depth_embed[lvl](feature_for_depth_reg)
            # 3D sizes
            # print("mushiyi>>>> ", self.dim3d_mode)
            if self.dim3d_mode == "from_mono":
                size3d = self.dim_embed_3d[lvl](hs_mono[lvl])
                outputs_3d_dims.append(size3d)
            elif self.dim3d_mode == "from_stereo":
                size3d = outputs_3d_dims[lvl]
            else:
                raise NotImplementedError
            scales =  img_sizes_ori[:, 1: 2] / (img_sizes[:, 1: 2] + img_sizes_upper.unsqueeze(1))
            box2d_height_norm = outputs_coord[:, :, 4] + outputs_coord[:, :, 5]
            box2d_height = torch.clamp(box2d_height_norm * img_sizes[:, 1: 2]*scales, min=1.0)
            depth_geo = size3d[:, :, 0] / box2d_height * calibs[:, 0, 0].unsqueeze(1) +(size3d[:, :, 2])/2
            
            # depth average + sigma
            if self.depth_pre_type == "avg":
                depth_ave = torch.cat([((1. / (depth_reg[:, :, 0: 1].sigmoid() + 1e-6) - 1.) \
                                        + depth_geo.unsqueeze(-1) \
                                            + depth_map) / 3,
                                    depth_reg[:, :, 1: 2]], -1)
            elif self.depth_pre_type == "avg2":
                depth_ave = torch.cat([((1. / (depth_reg[:, :, 0: 1].sigmoid() + 1e-6) - 1.) \
                                        + depth_map) / 2,
                                    depth_reg[:, :, 1: 2]], -1)
            elif self.depth_pre_type == "one_wo_uncertainty":
                depth_ave = torch.cat([depth_map,
                                    depth_reg[:, :, 1: 2]*0], -1)
            elif self.depth_pre_type == "detrhead_w_uncertainty":
                depth_ave = torch.cat([(1. / (depth_reg[:, :, 0: 1].sigmoid() + 1e-6) - 1.),
                                       depth_reg[:, :, 1: 2]], -1)
            elif self.depth_pre_type == "detrhead_wo_uncertainty":
                depth_ave = torch.cat([(1. / (depth_reg[:, :, 0: 1].sigmoid() + 1e-6) - 1.),
                                       depth_reg[:, :, 1: 2]*0], -1)
            elif self.depth_pre_type in[ "one_w_uncertainty"]:
                depth_ave = torch.cat([depth_map,
                                    depth_reg[:, :, 1: 2]], -1)
            elif self.depth_pre_type in ["sample_cat" ]:
                depth_ave = torch.cat([(1. / (depth_reg[:, :, 0: 1].sigmoid() + 1e-6) - 1.),
                                    depth_reg[:, :, 1: 2]], -1)
            elif self.depth_pre_type == "depthmap_and_geo_w_uncertainty":
                # comepare depth_map and depth_geo to get the max depth
                max_map_geo = (depth_map + depth_geo.unsqueeze(-1))/2
                depth_ave = torch.cat([depth_map_ref,
                                    depth_reg[:, :, 1: 2]], -1)
            elif self.depth_pre_type == "err_w_uncertainty":
                depth_ave = torch.cat([depth_map + depth_reg[:, :, 0: 1],
                                    depth_reg[:, :, 1: 2]], -1)
            else:
                raise NotImplementedError
            outputs_depths.append(depth_ave)
            # angles
            outputs_angle = self.angle_embed[lvl](hs_mono[lvl])
            outputs_angles.append(outputs_angle)
           
        outputs_coord = torch.stack(outputs_coords)
        outputs_samples = torch.stack(outputs_samples)
        outputs_class = torch.stack(outputs_classes)
        outputs_3d_dim = torch.stack(outputs_3d_dims)
        outputs_depth = torch.stack(outputs_depths)
        outputs_angle = torch.stack(outputs_angles)
        
  
        if self.Ood_det is not None:
            outputs_class_fg = inter_class
            prob_fg = outputs_class_fg[-1].sigmoid()

            prob_normals = outputs_class[-1][:,:,0:-1].sigmoid()
            prob_otherfg = outputs_class[-1][:,:,-1:].sigmoid()
            
            if self.Ood_det == "RBAF":
                prob_normals_mean = prob_normals.mean(dim=-1)
                Ood_score = prob_fg.squeeze(-1) - prob_normals_mean
                Ood_score = Ood_score.unsqueeze(-1)
            elif self.Ood_det == "MSP":
                prob_normals_softmax = F.softmax(outputs_class[-1][:,:,0:-1], dim=-1)
                Ood_score = prob_fg.squeeze(-1) - prob_normals_softmax.max(dim=-1)[0]
                Ood_score = Ood_score.unsqueeze(-1)
            elif self.Ood_det == "MaxLogit":
                Ood_score = prob_fg.squeeze(-1) -  outputs_class[-1][:,:,0:-1].max(dim=-1)[0]
                Ood_score = Ood_score.unsqueeze(-1)
            elif self.Ood_det == "MNPF":
                Ood_score = prob_fg.squeeze(-1) - prob_normals.max(dim=-1)[0]
                Ood_score = Ood_score.unsqueeze(-1)
            elif self.Ood_det == "JustOther":
                Ood_score = prob_otherfg
            else:
                Ood_score = None
        

        # print("debug pred_sample_points", outputs_samples[-1])
        # print("debug outputs_coord", outputs_coord[-1])
        out = {'pred_logits': outputs_class[-1],
                'pred_boxes': outputs_coord[-1],
                'pred_sample_points': outputs_samples[-1],
                }
        if self.Ood_det is not None:
            out['Ood_score'] = Ood_score
            out['pred_logits_fg'] = inter_classes[-1]
        out['pred_3d_dim'] = outputs_3d_dim[-1]
        out['pred_depth'] = outputs_depth[-1]
        out['pred_angle'] = outputs_angle[-1]
        out['pred_depth_map_logits'] = pred_depth_map_logits
        out['pred_disp'] = pred_disp
        
        out['inter_outputs'] = self._set_inter_loss(inter_class, inter_coords)
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth, outputs_samples)

        return out #, mask_dict

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth, outputs_samples):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 
                 'pred_3d_dim': c, 'pred_angle': d, 'pred_depth': e, 'pred_sample_points': f}
                for a, b, c, d, e ,f in zip(outputs_class[:-1], outputs_coord[:-1],
                                         outputs_3d_dim[:-1], outputs_angle[:-1], outputs_depth[:-1], outputs_samples[:-1])]
    @torch.jit.unused
    def _set_inter_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits_fg': a, 'pred_boxes': b}
                for a, b in zip(outputs_class, outputs_coord)]

class SetCriterion(nn.Module):
    """ This class computes the loss for MonoDETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, matcher_stereo, weight_dict, focal_alpha, losses, inter_losses,
                 group_num=11, depth_sort_reverse=False, depth_bg=0,
                 shrink_ratio=1, align_by_3d_center=False):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.matcher_stereo = matcher_stereo
        self.weight_dict = weight_dict
        self.losses = losses
        self.inter_losses = inter_losses
        self.focal_alpha = focal_alpha
        self.ddn_loss = DDNLoss(depth_sort_reverse=depth_sort_reverse, 
                                bg_value=depth_bg, 
                                shrink_ratio=shrink_ratio,
                                align_by_3d_center=align_by_3d_center,
                                )  # for depth map
        self.disparity_loss = DisparityLoss(maxdisp=96)
        self.loss_cls_2d = SigmoidFocalLoss(gamma=2.0, balance_weights=torch.tensor([4.0, 2.0, 4.0]))
        self.group_num = group_num
        self.depth_match_type =  "filter" # "filter" "rematch" 

    def loss_labels(self, outputs, targets, indices, indices_filted, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o.squeeze().long()

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses
    
    def loss_labels_fg(self, outputs, targets, indices, indices_filted, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        if 'pred_logits_fg' in outputs:
            src_logits_fg = outputs['pred_logits_fg']

            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            target_classes = torch.full(src_logits_fg.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits_fg.device)

            target_classes[idx] = target_classes_o.squeeze().long()
            # trans to fg 2 class label
            # 大于0的为前景，等于0的为背景
            target_classes = target_classes == self.num_classes 

            target_classes = target_classes.long() # 前景标注为0 背景为1

            target_classes_onehot = torch.zeros([src_logits_fg.shape[0], src_logits_fg.shape[1], src_logits_fg.shape[2]+1],
                                                dtype=src_logits_fg.dtype, layout=src_logits_fg.layout, device=src_logits_fg.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]
        
            loss_ce_fg = sigmoid_focal_loss(src_logits_fg, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits_fg.shape[1]
            losses = {'loss_ce_fg': loss_ce_fg}
            return losses
    
    def loss_labels_yolostesreo(self, outputs, targets, indices, indices_filted, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o.squeeze().long()

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        # loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        loss_ce = self.loss_cls_2d(src_logits, target_classes_onehot).sum() / src_logits.shape[1] 
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, indices_filted, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_3dcenter(self, outputs, targets, indices,  indices_filted, num_boxes):
        
        idx = self._get_src_permutation_idx(indices)
        src_3dcenter = outputs['pred_boxes'][:, :, 0: 2][idx]
        target_3dcenter = torch.cat([t['boxes_3d'][:, 0: 2][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_3dcenter = F.l1_loss(src_3dcenter, target_3dcenter, reduction='none')
        losses = {}
        losses['loss_center'] = loss_3dcenter.sum() / num_boxes
        return losses

    def loss_boxes(self, outputs, targets, indices, indices_filted, num_boxes):
        
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_2dboxes = outputs['pred_boxes'][:, :, 2: 6][idx]
        target_2dboxes = torch.cat([t['boxes_3d'][:, 2: 6][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # l1
        loss_bbox = F.l1_loss(src_2dboxes, target_2dboxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # giou
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcylrtb_to_xyxy(src_boxes),
            box_ops.box_cxcylrtb_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_depths_stable(self, outputs, targets, indices, indices_filted, num_boxes):  
        bs, nq = outputs['pred_depth'].shape[:2]

        # filter out low iou pairs
        indices_final = []
        if self.depth_match_type == "filter":
            indices_final = indices_filted
        elif self.depth_match_type == "rematch":
            for batch_index in range(bs):
                out_bbox_i = outputs["pred_boxes"][batch_index]  # [batch_size * num_queries, 4]
                tgt_bbox_i = targets[batch_index]["boxes_3d"]
                all_iou = box_iou(box_cxcylrtb_to_xyxy(out_bbox_i), box_cxcylrtb_to_xyxy(tgt_bbox_i))[0].view(nq, -1) # (b, num_queries, ngt) 
                indices_i = indices[batch_index]
                # max_iou, max_idx = all_iou.max(dim=1)
                if all_iou.shape[-1] > 0:
                    max_iou, max_idx = torch.max(all_iou, dim=1)
                    iou_mask = max_iou >= 0.5
                    positive_list = torch.nonzero(iou_mask, as_tuple=False)
                    if len(positive_list.shape) > 1:
                        positive_list = positive_list.squeeze(1)
                    max_idx = max_idx[iou_mask]
                    indices_final.append([positive_list.detach().cpu(), max_idx.detach().cpu()])
                else:
                    indices_final.append([torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)])
        else:
                print("must in rematch or filter ")
        idx = self._get_src_permutation_idx(indices_final)
        src_depths = outputs['pred_depth'][idx]
        target_depths = torch.cat([t['depth'][i] for t, (_, i) in zip(targets, indices_final)], dim=0).squeeze()

        depth_input, depth_log_variance = src_depths[:, 0], src_depths[:, 1] 
        _s = depth_input
 
        depth_loss = 1.4142 * torch.exp(-depth_log_variance) * torch.abs(depth_input - target_depths) + depth_log_variance  
        # depth_loss = F.l1_loss(depth_input, target_depths, reduction='none')
        losses = {}
        
        if len(idx[0]) > 0:
            losses['loss_depth'] = depth_loss.sum() / len(idx[0]) 
        else:
            losses['loss_depth'] = depth_loss.sum()*0.0
        return losses 

    def loss_depths(self, outputs, targets, indices, indices_filted, num_boxes):  

        idx = self._get_src_permutation_idx(indices)
   
        src_depths = outputs['pred_depth'][idx]
        target_depths = torch.cat([t['depth'][i] for t, (_, i) in zip(targets, indices)], dim=0).squeeze()

        depth_input, depth_log_variance = src_depths[:, 0], src_depths[:, 1] 
        depth_loss = 1.4142 * torch.exp(-depth_log_variance) * torch.abs(depth_input - target_depths) + depth_log_variance  
        # depth_loss = F.l1_loss(depth_input, target_depths, reduction='none')
        losses = {}
        losses['loss_depth'] = depth_loss.sum() / num_boxes 
        return losses 
    
    def loss_dims(self, outputs, targets, indices, indices_filted, num_boxes):  
        if self.depth_match_type == "filter":
            indices_final = indices_filted
        else:
            indices_final = indices
        idx = self._get_src_permutation_idx(indices_final)
        src_dims = outputs['pred_3d_dim'][idx]
        target_dims = torch.cat([t['size_3d'][i] for t, (_, i) in zip(targets, indices_final)], dim=0)

        dimension = target_dims.clone().detach()
        dim_loss = torch.abs(src_dims - target_dims)
        dim_loss /= dimension
        with torch.no_grad():
            compensation_weight = F.l1_loss(src_dims, target_dims) / dim_loss.mean()
        dim_loss *= compensation_weight
        losses = {}
        losses['loss_dim'] = dim_loss.sum() / len(idx[0]) 
        return losses

    def loss_angles(self, outputs, targets, indices, indices_filted, num_boxes):  
        if self.depth_match_type == "filter":
            indices_final = indices_filted
        else:
            indices_final = indices
        idx = self._get_src_permutation_idx(indices_final)
        heading_input = outputs['pred_angle'][idx]
        target_heading_cls = torch.cat([t['heading_bin'][i] for t, (_, i) in zip(targets, indices_final)], dim=0)
        target_heading_res = torch.cat([t['heading_res'][i] for t, (_, i) in zip(targets, indices_final)], dim=0)

        heading_input = heading_input.view(-1, 24)
        heading_target_cls = target_heading_cls.view(-1).long()
        heading_target_res = target_heading_res.view(-1)

        # classification loss
        heading_input_cls = heading_input[:, 0:12]
        cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction='none')

        # regression loss
        heading_input_res = heading_input[:, 12:24]
        cls_onehot = torch.zeros(heading_target_cls.shape[0], 12).cuda().scatter_(dim=1, index=heading_target_cls.view(-1, 1), value=1)
        heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
        reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction='none')
        
        angle_loss = cls_loss + reg_loss
        losses = {}
        losses['loss_angle'] = angle_loss.sum() / len(idx[0])  
        return losses

    def loss_depth_map(self, outputs, targets, indices, indices_filted, num_boxes):
        depth_map_logits = outputs['pred_depth_map_logits']
        _, _, H_depth_map, W_depth_map = depth_map_logits.shape
        num_gt_per_img = [len(t['boxes']) for t in targets]
        gt_boxes2d = torch.cat([t['boxes'] for t in targets], dim=0) * torch.tensor([W_depth_map, H_depth_map, W_depth_map, H_depth_map], device='cuda')
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
        gt_center_depth = torch.cat([t['depth'] for t in targets], dim=0).squeeze(dim=1)
        gt_boxes3d = torch.cat([t['boxes_3d'] for t in targets], dim=0)* \
            torch.tensor([W_depth_map, H_depth_map, W_depth_map, W_depth_map, H_depth_map, H_depth_map], device='cuda')
        
        losses = dict()

        losses["loss_depth_map"] = self.ddn_loss(
            depth_map_logits, gt_boxes2d, gt_boxes3d, num_gt_per_img, gt_center_depth)
        return losses
    
    def loss_disp_map(self, outputs, targets, indices, indices_filted, num_boxes):
        disp_map_pre = outputs['pred_disp']
        disp_maps = torch.cat([t['disp'] for t in targets], dim=0).squeeze(dim=1)
        losses = dict()
        losses["loss_disp_map"] = self.disparity_loss(disp_map_pre, disp_maps)
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, indices_filted, num_boxes, **kwargs):
        
        loss_map = {
            'labels': self.loss_labels,
            'labels_fg': self.loss_labels_fg,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'depths': self.loss_depths_stable,
            'dims': self.loss_dims,
            'angles': self.loss_angles,
            'center': self.loss_3dcenter,
            'depth_map': self.loss_depth_map,
            'disp_map': self.loss_disp_map,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, indices_filted, num_boxes, **kwargs)

    def forward(self, outputs, targets, mask_dict=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        
        group_num = self.group_num if self.training else 1
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets) * group_num
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        losses = {}

        # Compute Det fg loss
        for i, inter_outputs in enumerate(outputs['inter_outputs']):
            indices, indices_filted = self.matcher(inter_outputs, targets, group_num=group_num, match_mode="stereo")
            for loss in self.inter_losses:
                l_dict = self.get_loss(loss, inter_outputs, targets, indices,indices_filted, num_boxes)
                l_dict = {k + f'_inter_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
                
                
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'inter_outputs'}
        group_num = self.group_num if self.training else 1

        # Retrieve the matching between the outputs of the last layer and the targets
        indices, indices_filted = self.matcher(outputs_without_aux, targets, group_num=group_num, match_mode="mono")
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # num_boxes = sum(len(t["labels"]) for t in targets) * group_num
        # num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        for loss in self.losses:
            #ipdb.set_trace()
            losses.update(self.get_loss(loss, outputs, targets, indices, indices_filted, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices, indices_filted = self.matcher(aux_outputs, targets, group_num=group_num)
                for loss in self.losses:

                    if loss in ['depth_map', 'labels_fg', 'disp_map', 'dims','depths', 'angles', 'center', 'sample_point']:
                    # if loss in ['depth_map', 'labels_fg', 'disp_map']:
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                        
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, indices_filted, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_lightstereoOod(cfg):
    # backbone
    backbone = build_backbone(cfg)

    # detr
    det_stereo_transformer = build_det_stereo_transformer(cfg)
    det_mono_transformer = build_det_mono_transformer(cfg)
    # depth prediction module
    depth_predictor = LightStereoDepthPredictor(cfg)
    model = StereoLightDETROoD(
        backbone,
        det_stereo_transformer,
        det_mono_transformer, 
        depth_predictor,
        num_classes=cfg['num_classes'],
        num_queries=cfg['num_queries'],
        aux_loss=cfg['aux_loss'],
        num_feature_levels=cfg['num_feature_levels'],
        with_box_refine=cfg['with_box_refine'],
        with_center_refine=cfg['with_center_refine'],
        init_box=cfg['init_box'],
        group_num=cfg['group_num'],
        depth_pre_type=cfg['depth_pre_type'],
        share_Q=cfg['share_Q'],
        depth_sample_mode=cfg['depth_sample_mode'],
        fusion_mode=cfg['fusion_mode'],
        num_scale=cfg['num_scale'],
        FPN_type=cfg['FPN_type'],
        Ood_det=cfg['Ood_det'],
        dim3d_mode=cfg['dim3d_mode'])

    # matcher
    matcher = build_fg_Stablematcher(cfg)
    # matcher
    matcher_stereo = build_matcher(cfg)
    # loss
    weight_dict = {'loss_ce': cfg['cls_loss_coef'], 'loss_bbox': cfg['bbox_loss_coef']}
    weight_dict['loss_ce_fg'] = cfg['cls_loss_coef_fg']
    weight_dict['loss_giou'] = cfg['giou_loss_coef']
    weight_dict['loss_dim'] = cfg['dim_loss_coef']
    weight_dict['loss_angle'] = cfg['angle_loss_coef']
    weight_dict['loss_depth'] = cfg['depth_loss_coef']
    weight_dict['loss_center'] = cfg['3dcenter_loss_coef']
    weight_dict['loss_depth_map'] = cfg['depth_map_loss_coef']
    weight_dict['loss_disp_map'] = cfg['disp_map_loss_coef']
    
    # dn loss
    if cfg['use_dn']:
        weight_dict['tgt_loss_ce']= cfg['cls_loss_coef']
        weight_dict['tgt_loss_bbox'] = cfg['bbox_loss_coef']
        weight_dict['tgt_loss_giou'] = cfg['giou_loss_coef']
        weight_dict['tgt_loss_angle'] = cfg['angle_loss_coef']
        weight_dict['tgt_loss_center'] = cfg['3dcenter_loss_coef']

    # TODO this is a hack
    if cfg['aux_loss']:
        aux_weight_dict = {}
        for i in range(cfg['dec_layers'] - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    inter_weight_dict = {}
    inter_keys = ['loss_ce_fg', 'loss_bbox', 'loss_center', 'loss_giou']
    layers = cfg['dec_layers']
    for i in range(layers):
        inter_weight_dict.update({k + f'_inter_{i}': v for k, v in weight_dict.items() if k in inter_keys})
    weight_dict.update(inter_weight_dict)
            
    inter_losses = ['labels_fg', 'boxes', 'center']
    
    if cfg['Ood_det'] is not None:
        losses = ['labels', 'labels_fg', 'boxes', 'cardinality', 'depths', 'dims', 'angles', 'center', 'depth_map', 'disp_map']
    else:
        losses = ['labels', 'boxes', 'cardinality', 'depths', 'dims', 'angles', 'center', 'depth_map', 'disp_map']
    
    criterion = SetCriterion(
        cfg['num_classes'],
        matcher=matcher,
        matcher_stereo = matcher_stereo,
        weight_dict=weight_dict,
        focal_alpha=cfg['focal_alpha'],
        losses=losses,
        inter_losses=inter_losses,
        group_num=cfg['group_num'],
        depth_sort_reverse=cfg['depth_sort_reverse'],
        depth_bg=cfg['depth_bg'],
        shrink_ratio=cfg['shrink_ratio'], 
        align_by_3d_center=cfg['align_by_3d_center'])

    device = torch.device(cfg['device'])
    criterion.to(device)
    
    return model, criterion
