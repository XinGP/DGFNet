from typing import Any, Dict, List, Tuple, Union, Optional
import time
import math
import numpy as np
from fractions import gcd

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention

from utils.utils import gpu, init_weights


class DGFNet(nn.Module):
    # Initialization
    def __init__(self, cfg, device):
        super(DGFNet, self).__init__()
        self.device = device
        
        self.g_num_modes = 6
        self.g_pred_len = 30
        self.g_num_coo = 2
        self.d_embed = 128

        self.actor_net_ac = Actor_Encoder(device=self.device,
                                  n_in=cfg['in_actor'],
                                  hidden_size=cfg['d_actor'],
                                  n_fpn_scale=cfg['n_fpn_scale'])

        self.actor_net_sc = Actor_Encoder(device=self.device,
                                  n_in=cfg['in_actor'],
                                  hidden_size=cfg['d_actor'],
                                  n_fpn_scale=cfg['n_fpn_scale'])

        self.lane_net_ac = Map_Encoder(device=self.device,
                                in_size=cfg['in_lane'],
                                hidden_size=cfg['d_lane'],
                                dropout=cfg['dropout'])

        self.lane_net_sc = Map_Encoder(device=self.device,
                                in_size=cfg['in_lane_sc'],
                                hidden_size=cfg['d_lane'],
                                dropout=cfg['dropout'])

        self.ac_am_fusion = FusionNet(device=self.device,
                                    config=cfg)

        self.interaction_module_sc_am = Interaction_Module_SC_AM(device=self.device,
                                                               hidden_size = cfg['d_lane'],
                                                               depth = cfg["n_interact"])

        self.interaction_module_af = Interaction_Module_FE(device=self.device,
                                                         hidden_size = cfg['d_lane'],
                                                         depth = cfg["n_interact"])

        self.interaction_module_al = Interaction_Module_FE(device=self.device,
                                                         hidden_size = cfg['d_lane'],
                                                         depth = cfg["n_interact"])

        self.trajectory_decoder_fe = Trajectory_Decoder_Future_Enhanced(device=self.device,
                                                                       hidden_size = cfg['d_lane'])

        self.trajectory_decoder_final = Trajectory_Decoder_Final(device=self.device,
                                                                hidden_size = cfg["d_decoder_F"])

        self.rft_encoder = Reliable_Future_Trajectory_Encoder()

        if cfg["init_weights"]:
            self.apply(init_weights)

    def forward(self, data):
        actors_ac, actor_idcs, lanes_ac, lane_idcs, rpe,  actors_sc, lanes_sc = data

        batch_size = len(actor_idcs)

        # ac actors/lanes encoding
        actors_ac = self.actor_net_ac(actors_ac)

        lanes_ac = self.lane_net_ac(lanes_ac)

        # ac feature fusion
        actors_ac, _ , _ = self.ac_am_fusion(actors_ac, actor_idcs, lanes_ac, lane_idcs, rpe)

        # sc actors/lanes encoding
        actors_sc = self.actor_net_sc(actors_sc)
        lanes_sc = self.lane_net_sc(lanes_sc)

        agent_lengths = []
        for actor_id in actor_idcs:
            agent_lengths.append(actor_id.shape[0] if actor_id is not None else 0)

        lane_lengths = []
        for lane_id in lane_idcs:
            lane_lengths.append(lane_id.shape[0] if lane_id is not None else 0)
        
        max_agent_num = max(agent_lengths)
        max_lane_num = max(lane_lengths)

        actors_batch_sc = torch.zeros(batch_size, max_agent_num, self.d_embed, device=self.device)
        actors_batch_ac = torch.zeros(batch_size, max_agent_num, self.d_embed, device=self.device)
        lanes_batch_sc = torch.zeros(batch_size, max_lane_num, self.d_embed, device=self.device)

        for i, actor_ids in enumerate(actor_idcs):
            num_agents = actor_ids.shape[0]
            actors_batch_sc[i, :num_agents] = actors_sc[actor_ids[0] : actor_ids[-1] + 1]
            actors_batch_ac[i, :num_agents] = actors_ac[actor_ids[0] : actor_ids[-1] + 1]

        for i, lane_ids in enumerate(lane_idcs):
            num_lanes = lane_ids.shape[0]
            lanes_batch_sc[i, :num_lanes] = lanes_sc[lane_ids[0] : lane_ids[-1] + 1]
        
        masks, _ = get_masks(agent_lengths, lane_lengths, self.device)
        agent_states, lane_states = self.interaction_module_sc_am(actors_batch_sc, lanes_batch_sc, masks)

        #reliable future trajectory generate
        predictions_fe = self.trajectory_decoder_fe(agent_states)
        
        final_positions = predictions_fe[:, :, :, -1, :]

        mean_final_positions = final_positions.mean(dim=2)

        deviations = torch.sqrt(((final_positions - mean_final_positions.unsqueeze(2)) ** 2).sum(dim=-1))

        mean_deviation = deviations.mean(dim=-1)
        mask = (mean_deviation <= 5).float()

        mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        masked_predictions = predictions_fe * mask

        future_states = self.rft_encoder(masked_predictions)

        #future feature interaction
        masks_af = get_mask(agent_lengths, agent_lengths, self.device)

        actors_batch_af, _ = self.interaction_module_af(actors_batch_ac, future_states, masks_af)

        masks_al = get_mask(agent_lengths, lane_lengths, self.device)

        actors_batch_al, _ = self.interaction_module_al(actors_batch_af, lanes_batch_sc, masks_al)

        agent_fusion = torch.cat((actors_batch_al, agent_states), dim = 2)

        predictions_final, logits_final = self.trajectory_decoder_final(agent_fusion)

        expan_predictions_fe = torch.zeros((actors_sc.shape[0], self.g_num_modes, self.g_pred_len, self.g_num_coo), device=self.device)
        expan_predictions_final = torch.zeros((actors_sc.shape[0], self.g_num_modes, self.g_pred_len, self.g_num_coo), device=self.device)
        expan_logits_final = torch.zeros((actors_sc.shape[0], self.g_num_modes), device=self.device)

        for i, actor_ids in enumerate(actor_idcs):
            num_agents = actor_ids.shape[0]
            expan_predictions_fe[actor_ids[0]:actor_ids[-1] + 1] = predictions_fe[i, :num_agents]
            expan_predictions_final[actor_ids[0]:actor_ids[-1] + 1] = predictions_final[i, :num_agents]
            expan_logits_final[actor_ids[0]:actor_ids[-1] + 1] = logits_final[i, :num_agents]

        res_reg_fe, res_reg_final , res_cls_final = [], [], []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            res_reg_fe.append(expan_predictions_fe[idcs])
            res_reg_final.append(expan_predictions_final[idcs])
            res_cls_final.append(expan_logits_final[idcs])
        
        out_sc = [res_cls_final, res_reg_final, res_reg_fe]

        return out_sc

    def pre_process(self, data):
        '''
            Send to device
            'BATCH_SIZE', 'SEQ_ID', 'CITY_NAME',
            'ORIG', 'ROT',
            'TRAJS_OBS', 'TRAJS_FUT', 'PAD_OBS', 'PAD_FUT', 'TRAJS_CTRS', 'TRAJS_VECS',
            'LANE_GRAPH',
            'RPE',
            'ACTORS', 'ACTOR_IDCS', 'LANES', 'LANE_IDCS','LANES_SC','ACTORS_SC'
        '''
        actors = gpu(data['ACTORS'], self.device)
        actors_sc = gpu(data['ACTORS_SC'], self.device)
        actor_idcs = gpu(data['ACTOR_IDCS'], self.device)
        lanes = gpu(data['LANES'], self.device)
        lanes_sc = gpu(data['LANES_SC'], self.device)
        lane_idcs = gpu(data['LANE_IDCS'], self.device)
        rpe = gpu(data['RPE'], self.device)

        return actors, actor_idcs, lanes, lane_idcs, rpe , actors_sc, lanes_sc

    def post_process(self, out):
        post_out = dict()
        res_cls = out[0]
        res_reg = out[1]

        # get prediction results for target vehicles only
        reg = torch.stack([trajs[0] for trajs in res_reg], dim=0)
        cls = torch.stack([probs[0] for probs in res_cls], dim=0)

        post_out['out_sc'] = out
        post_out['traj_pred'] = reg  # batch x n_mod x pred_len x 2
        post_out['prob_pred'] = cls  # batch x n_mod

        return post_out

class Conv1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Conv1d, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.conv = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, padding=(
            int(kernel_size) - 1) // 2, stride=stride, bias=False)

        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out

class Res1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Res1d, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size=kernel_size, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # All use name bn1 and bn2 to load imagenet pretrained weights
        if norm == 'GN':
            self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.bn1 = nn.BatchNorm1d(n_out)
            self.bn2 = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        if stride != 1 or n_out != n_in:
            if norm == 'GN':
                self.downsample = nn.Sequential(
                    nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.downsample = nn.Sequential(
                    nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(n_out))
            else:
                exit('SyncBN has not been added!')
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out


class Actor_Encoder(nn.Module):
    """
    Actor feature extractor with Conv1D
    """

    def __init__(self, device, n_in=3, hidden_size=128, n_fpn_scale=4):
        super(Actor_Encoder, self).__init__()
        self.device = device
        norm = "GN"
        ng = 1

        n_out = [2**(5 + s) for s in range(n_fpn_scale)]  # [32, 64, 128]
        blocks = [Res1d] * n_fpn_scale
        num_blocks = [2] * n_fpn_scale

        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], hidden_size, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)

        self.output = Res1d(hidden_size, hidden_size, norm=norm, ng=ng)

    def forward(self, actors: Tensor) -> Tensor:
        out = actors

        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)
            out += self.lateral[i](outputs[i])

        out = self.output(out)[:, :, -1]

        return out

class PointFeatureAggregator(nn.Module):
    def __init__(self, hidden_size: int, aggre_out: bool, dropout: float = 0.1) -> None:
        super(PointFeatureAggregator, self).__init__()
        self.aggre_out = aggre_out

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.norm = nn.LayerNorm(hidden_size)

    def _global_maxpool_aggre(self, feat):
        return F.adaptive_max_pool1d(feat.permute(0, 2, 1), 1).permute(0, 2, 1)

    def forward(self, x_inp):
        x = self.fc1(x_inp)  # [N_{lane}, 10, hidden_size]
        x_aggre = self._global_maxpool_aggre(x)
        x_aggre = torch.cat([x, x_aggre.repeat([1, x.shape[1], 1])], dim=-1)

        out = self.norm(x_inp + self.fc2(x_aggre))
        if self.aggre_out:
            return self._global_maxpool_aggre(out).squeeze()
        else:
            return out


class Map_Encoder(nn.Module):
    def __init__(self, device, in_size=10, hidden_size=128, dropout=0.1):
        super(Map_Encoder, self).__init__()
        self.device = device

        self.proj = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.aggre1 = PointFeatureAggregator(hidden_size=hidden_size, aggre_out=False, dropout=dropout)
        self.aggre2 = PointFeatureAggregator(hidden_size=hidden_size, aggre_out=True, dropout=dropout)
    '''
    def forward(self, feats):
        outs = []
        for feat in feats:
            x = self.proj(feat)  # [N_{lane}, 10, hidden_size]
            x = self.aggre1(x)
            x = self.aggre2(x)  # [N_{lane}, hidden_size]
            outs.extend(x)
        return outs
    '''
    def forward(self, feats):
        x = self.proj(feats)  # [N_{lane}, 10, hidden_size]
        x = self.aggre1(x)
        x = self.aggre2(x)  # [N_{lane}, hidden_size]
        return x

class Spatial_Feature_Layer(nn.Module):
    def __init__(self,
                 device,
                 d_edge: int = 128,
                 d_model: int = 128,
                 d_ffn: int = 2048,
                 n_head: int = 8,
                 dropout: float = 0.1,
                 update_edge: bool = True) -> None:
        super(Spatial_Feature_Layer, self).__init__()
        self.device = device
        self.update_edge = update_edge

        self.proj_memory = nn.Sequential(
            nn.Linear(d_model + d_model + d_edge, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True)
        )

        if self.update_edge:
            self.proj_edge = nn.Sequential(
                nn.Linear(d_model, d_edge),
                nn.LayerNorm(d_edge),
                nn.ReLU(inplace=True)
            )
            self.norm_edge = nn.LayerNorm(d_edge)

        self.multihead_attn = MultiheadAttention(
            embed_dim=d_model, num_heads=n_head, dropout=dropout, batch_first=False)

        # Feedforward model
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self,
                node: Tensor,
                edge: Tensor,
                edge_mask: Optional[Tensor]) -> Tensor:
        '''
            input:
                node:       (N, d_model)
                edge:       (N, N, d_model)
                edge_mask:  (N, N)
        '''
        # update node
        x, edge, memory = self._build_memory(node, edge)
        x_prime, _ = self._mha_block(x, memory, attn_mask=None, key_padding_mask=edge_mask)
        x = self.norm2(x + x_prime).squeeze()
        x = self.norm3(x + self._ff_block(x))
        return x, edge, None

    def _build_memory(self,
                      node: Tensor,
                      edge: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        '''
            input:
                node:   (N, d_model)
                edge:   (N, N, d_edge)
            output:
                :param  (1, N, d_model)
                :param  (N, N, d_edge)
                :param  (N, N, d_model)
        '''
        n_token = node.shape[0]

        # 1. build memory
        src_x = node.unsqueeze(dim=0).repeat([n_token, 1, 1])  # (N, N, d_model)
        tar_x = node.unsqueeze(dim=1).repeat([1, n_token, 1])  # (N, N, d_model)
        memory = self.proj_memory(torch.cat([edge, src_x, tar_x], dim=-1))  # (N, N, d_model)
        # 2. (optional) update edge (with residual)
        if self.update_edge:
            edge = self.norm_edge(edge + self.proj_edge(memory))  # (N, N, d_edge)

        return node.unsqueeze(dim=0), edge, memory

    # multihead attention block
    def _mha_block(self,
                   x: Tensor,
                   mem: Tensor,
                   attn_mask: Optional[Tensor],
                   key_padding_mask: Optional[Tensor]) -> Tensor:
        '''
            input:
                x:                  [1, N, d_model]
                mem:                [N, N, d_model]
                attn_mask:          [N, N]
                key_padding_mask:   [N, N]
            output:
                :param      [1, N, d_model]
                :param      [N, N]
        '''
        x, _ = self.multihead_attn(x, mem, mem,
                                   attn_mask=attn_mask,
                                   key_padding_mask=key_padding_mask,
                                   need_weights=False)  # return average attention weights
        return self.dropout2(x), None

    # feed forward block
    def _ff_block(self,
                  x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class SymmetricFusionTransformer(nn.Module):
    def __init__(self,
                 device,
                 d_model: int = 128,
                 d_edge: int = 128,
                 n_head: int = 8,
                 n_layer: int = 6,
                 dropout: float = 0.1,
                 update_edge: bool = True):
        super(SymmetricFusionTransformer, self).__init__()
        self.device = device

        fusion = []
        for i in range(n_layer):
            need_update_edge = False if i == n_layer - 1 else update_edge
            fusion.append(Spatial_Feature_Layer(device=device,
                                   d_edge=d_edge,
                                   d_model=d_model,
                                   d_ffn=d_model*2,
                                   n_head=n_head,
                                   dropout=dropout,
                                   update_edge=need_update_edge))
        self.fusion = nn.ModuleList(fusion)

    def forward(self, x: Tensor, edge: Tensor, edge_mask: Tensor) -> Tensor:
        '''
            x: (N, d_model)
            edge: (d_model, N, N)
            edge_mask: (N, N)
        '''
        # attn_multilayer = []
        for mod in self.fusion:
            x, edge, _ = mod(x, edge, edge_mask)
            # attn_multilayer.append(attn)
        return x, None


class FusionNet(nn.Module):
    def __init__(self, device, config):
        super(FusionNet, self).__init__()
        self.device = device

        d_embed = config['d_embed']
        dropout = config['dropout']
        update_edge = config['update_edge']

        self.proj_actor = nn.Sequential(
            nn.Linear(config['d_actor'], d_embed),
            nn.LayerNorm(d_embed),
            nn.ReLU(inplace=True)
        )
        self.proj_lane = nn.Sequential(
            nn.Linear(config['d_lane'], d_embed),
            nn.LayerNorm(d_embed),
            nn.ReLU(inplace=True)
        )
        self.proj_rpe_scene = nn.Sequential(
            nn.Linear(config['d_rpe_in'], config['d_rpe']),
            nn.LayerNorm(config['d_rpe']),
            nn.ReLU(inplace=True)
        )

        self.fuse_scene = SymmetricFusionTransformer(self.device,
                                                     d_model=d_embed,
                                                     d_edge=config['d_rpe'],
                                                     n_head=config['n_scene_head'],
                                                     n_layer=config['n_scene_layer'],
                                                     dropout=dropout,
                                                     update_edge=update_edge)

    def forward(self,
                actors: Tensor,
                actor_idcs: List[Tensor],
                lanes: Tensor,
                lane_idcs: List[Tensor],
                rpe_prep: Dict[str, Tensor]):
        # print('actors: ', actors.shape)
        # print('actor_idcs: ', [x.shape for x in actor_idcs])
        # print('lanes: ', lanes.shape)
        # print('lane_idcs: ', [x.shape for x in lane_idcs])

        # projection
        actors = self.proj_actor(actors)
        lanes = self.proj_lane(lanes)

        actors_new, lanes_new = list(), list()
        for a_idcs, l_idcs, rpes in zip(actor_idcs, lane_idcs, rpe_prep):
            # * fusion - scene
            _actors = actors[a_idcs]
            _lanes = lanes[l_idcs]
            tokens = torch.cat([_actors, _lanes], dim=0)  # (N, d_model)
            rpe = self.proj_rpe_scene(rpes['scene'].permute(1, 2, 0))  # (N, N, d_rpe)
            out, _ = self.fuse_scene(tokens, rpe, rpes['scene_mask'])

            actors_new.append(out[:len(a_idcs)])
            lanes_new.append(out[len(a_idcs):])
        # print('actors: ', [x.shape for x in actors_new])
        # print('lanes: ', [x.shape for x in lanes_new])
        actors = torch.cat(actors_new, dim=0)
        lanes = torch.cat(lanes_new, dim=0)
        # print('actors: ', actors.shape)
        # print('lanes: ', lanes.shape)
        return actors, lanes, None

class Interaction_Module_SC_AM(nn.Module):
    def __init__(self, device, hidden_size, depth=3):
        super(Interaction_Module_SC_AM, self).__init__()

        self.depth = depth

        self.AA = nn.ModuleList([Attention_Block(hidden_size) for _ in range(depth)])
        self.AL = nn.ModuleList([Attention_Block(hidden_size) for _ in range(depth)])
        self.LL = nn.ModuleList([Attention_Block(hidden_size) for _ in range(depth)])
        self.LA = nn.ModuleList([Attention_Block(hidden_size) for _ in range(depth)])

    def forward(self, agent_features, lane_features, masks):

        for layer_index in range(self.depth):
            # === Lane to Agent ===
            lane_features = self.LA[layer_index](lane_features, agent_features, attn_mask=masks[-1])
            # === === ===

            # === Lane to Lane ===
            lane_features = self.LL[layer_index](lane_features, attn_mask=masks[-2])
            # === ==== ===

            # === Agent to Lane ===
            agent_features = self.AL[layer_index](agent_features, lane_features, attn_mask=masks[-3])
            # === ==== ===

            # === Agent to Agent ===
            agent_features = self.AA[layer_index](agent_features, attn_mask=masks[-4])
            # === ==== ===

        return agent_features, lane_features


class Interaction_Module_FE(nn.Module):
    def __init__(self, device, hidden_size, depth=3):
        super(Interaction_Module_FE, self).__init__()

        self.depth = depth
        self.AL = nn.ModuleList([Attention_Block(hidden_size) for _ in range(depth)])


    def forward(self, agent_features, lane_features, masks):

        for layer_index in range(self.depth):

            # === Agent to Lane ===
            agent_features = self.AL[layer_index](agent_features, lane_features, attn_mask=masks[0])
            # === ==== ===

        return agent_features, lane_features


class Attention_Block(nn.Module):
    def __init__(self, hidden_size, num_heads=8, p_drop=0.1):

        super(Attention_Block, self).__init__()
        self.multiheadattention = Attention(hidden_size, num_heads, p_drop)

        self.ffn_layer = MLP(hidden_size, hidden_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, query, key_value=None, attn_mask=None):
        if key_value is None:
            key_value = query

        attn_output = self.multiheadattention(
            query, key_value, attention_mask=attn_mask)

        query = self.norm1(attn_output + query)
        query_temp = self.ffn_layer(query)
        query = self.norm2(query_temp + query)

        return query


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, p_drop=0.0, hidden_dim=None, residual=False):
        super(MLP, self).__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        layer2_dim = hidden_dim
        if residual:
            layer2_dim = hidden_dim + input_dim

        self.residual = residual
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(layer2_dim, output_dim)
        self.dropout1 = nn.Dropout(p=p_drop)
        self.dropout2 = nn.Dropout(p=p_drop)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout1(out)
        if self.residual:
            out = self.layer2(torch.cat([out, x], dim=-1))
        else:
            out = self.layer2(out)

        out = self.dropout2(out)
        return out

class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, p_drop):
        super(Attention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.last_projection = nn.Linear(self.all_head_size, hidden_size)
        self.attention_drop = nn.Dropout(p_drop)

    def get_extended_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def transpose_for_scores(self, x):
        sz = x.size()[:-1] + (self.num_attention_heads,
                              self.attention_head_size)
        # (batch, max_vector_num, head, head_size)
        x = x.view(*sz)
        # (batch, head, max_vector_num, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_states, key_value_states, attention_mask):
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = F.linear(key_value_states, self.key.weight)
        mixed_value_layer = self.value(key_value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(
            query_layer/math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))

        if attention_mask is not None:
            attention_scores = attention_scores + \
                self.get_extended_attention_mask(attention_mask)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attention_drop(attention_probs)

        assert torch.isnan(attention_probs).sum() == 0

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        # context_layer.shape = (batch, max_vector_num, all_head_size)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.last_projection(context_layer)
        return context_layer

class Trajectory_Decoder_Future_Enhanced(nn.Module):
    def __init__(self, device, hidden_size):
        super(Trajectory_Decoder_Future_Enhanced, self).__init__()
        self.endpoint_predictor = MLP(hidden_size, 6*2, residual=True)

        self.get_trajectory = MLP(hidden_size + 2, 29*2, residual=True)
        self.endpoint_refiner = MLP(hidden_size + 2, 2, residual=True)
        #self.get_prob = MLP(hidden_size + 2, 1, residual=True)

    def forward(self, agent_features):
        # agent_features.shape = (N, M, 128)
        N = agent_features.shape[0]
        M = agent_features.shape[1]
        D = agent_features.shape[2]

        # endpoints.shape = (N, M, 6, 2)
        endpoints = self.endpoint_predictor(agent_features).view(N, M, 6, 2)

        # prediction_features.shape = (N, M, 6, 128)
        agent_features_expanded = agent_features.unsqueeze(dim=2).expand(N, M, 6, D)

        # offsets.shape = (N, M, 6, 2)
        offsets = self.endpoint_refiner(torch.cat([agent_features_expanded, endpoints.detach()], dim=-1))
        endpoints += offsets

        # agent_features_expanded.shape = (N, M, 6, 128 + 2)
        agent_features_expanded = torch.cat([agent_features_expanded, endpoints.detach()], dim=-1)

        predictions = self.get_trajectory(agent_features_expanded).view(N, M, 6, 29, 2)
        #logits = self.get_prob(agent_features_expanded).view(N, M, 6)

        predictions = torch.cat([predictions, endpoints.unsqueeze(dim=-2)], dim=-2)

        assert predictions.shape == (N, M, 6, 30, 2)

        return predictions

class Trajectory_Decoder_Final(nn.Module):
    def __init__(self, device, hidden_size):
        super(Trajectory_Decoder_Final, self).__init__()
        self.endpoint_predictor = MLP(hidden_size, 6*2, residual=True)

        self.get_trajectory = MLP(hidden_size + 2, 29*2, residual=True)
        self.endpoint_refiner = MLP(hidden_size + 2, 2, residual=True)

        self.cls = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size + 2),
            nn.LayerNorm(hidden_size+ 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size + 2, hidden_size + 2),
            nn.LayerNorm(hidden_size + 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size + 2, 1)
        )

    def forward(self, agent_features):
        # agent_features.shape = (N, M, 128)
        N = agent_features.shape[0]
        M = agent_features.shape[1]
        D = agent_features.shape[2]

        # endpoints.shape = (N, M, 6, 2)
        endpoints = self.endpoint_predictor(agent_features).view(N, M, 6, 2)

        # prediction_features.shape = (N, M, 6, 128)
        agent_features_expanded = agent_features.unsqueeze(dim=2).expand(N, M, 6, D)

        # offsets.shape = (N, M, 6, 2)
        offsets = self.endpoint_refiner(torch.cat([agent_features_expanded, endpoints.detach()], dim=-1))
        endpoints += offsets

        # agent_features_expanded.shape = (N, M, 6, 128 + 2)
        agent_features_expanded = torch.cat([agent_features_expanded, endpoints.detach()], dim=-1)

        predictions = self.get_trajectory(agent_features_expanded).view(N, M, 6, 29, 2)

        #logits = self.get_prob(agent_features_expanded).view(N, M, 6)

        logits = self.cls(agent_features_expanded).view(N, M, 6)
        logits = F.softmax(logits * 1.0, dim=2)  # e.g., [159, 6]

        predictions = torch.cat([predictions, endpoints.unsqueeze(dim=-2)], dim=-2)

        assert predictions.shape == (N, M, 6, 30, 2)

        return predictions, logits

class Reliable_Future_Trajectory_Encoder(nn.Module):
    def __init__(self):
        super(Reliable_Future_Trajectory_Encoder, self).__init__()
        self.get_encoder = MLP(360, 128, residual=True)

    def forward(self, agent_features):
        # agent_features.shape = (N, M, 128)
        N, M, _, _, _ = agent_features.shape

        flattened_input = agent_features.view(N, M, -1)

        future_features = self.get_encoder(flattened_input).view(N, M, 128)

        return future_features

def get_mask(agent_lengths, lane_lengths, device):
    max_lane_num = max(lane_lengths)
    max_agent_num = max(agent_lengths)
    batch_size = len(agent_lengths)

    # === Agent - Lane Mask ===
    # query: agent, key-value: lane
    AL_mask = torch.zeros(
        batch_size, max_agent_num, max_lane_num, device=device)

    for i, (agent_length, lane_length) in enumerate(zip(agent_lengths, lane_lengths)):
        AL_mask[i, :agent_length, :lane_length] = 1

    masks = [AL_mask]

    # === === === === ===
    return masks

def get_masks(agent_lengths, lane_lengths, device):
    max_lane_num = max(lane_lengths)
    max_agent_num = max(agent_lengths)
    batch_size = len(agent_lengths)

    # === === Mask Generation Part === ===
    # === Agent - Agent Mask ===
    # query: agent, key-value: agent
    AA_mask = torch.zeros(
        batch_size, max_agent_num, max_agent_num, device=device)

    for i, agent_length in enumerate(agent_lengths):
        AA_mask[i, :agent_length, :agent_length] = 1
    # === === ===

    # === Agent - Lane Mask ===
    # query: agent, key-value: lane
    AL_mask = torch.zeros(
        batch_size, max_agent_num, max_lane_num, device=device)

    for i, (agent_length, lane_length) in enumerate(zip(agent_lengths, lane_lengths)):
        AL_mask[i, :agent_length, :lane_length] = 1
    # === === ===

    # === Lane - Lane Mask ===
    # query: lane, key-value: lane
    LL_mask = torch.zeros(
        batch_size, max_lane_num, max_lane_num, device=device)

    QL_mask = torch.zeros(
        batch_size, 6, max_lane_num, device=device)

    for i, lane_length in enumerate(lane_lengths):
        LL_mask[i, :lane_length, :lane_length] = 1

        QL_mask[i, :, :lane_length] = 1

    # === === ===

    # === Lane - Agent Mask ===
    # query: lane, key-value: agent
    LA_mask = torch.zeros(
        batch_size, max_lane_num, max_agent_num, device=device)

    for i, (lane_length, agent_length) in enumerate(zip(lane_lengths, agent_lengths)):
        LA_mask[i, :lane_length, :agent_length] = 1

    # === === ===

    masks = [AA_mask, AL_mask, LL_mask, LA_mask]

    # === === === === ===

    return masks, QL_mask
