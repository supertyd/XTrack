import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

from lib.models.layers.attn import Attention
from lib.models.layers.adapter import Bi_direct_adapter

from lib.models.layers.moe_lora import MoE_lora

class Prompt_block(nn.Module, ):
    def __init__(self, inplanes=None, mlp_rate=None, smooth=False,patch_num =None):
        super(Prompt_block, self).__init__()
        hidden = 64

        self.ffn0_1 = MoE_lora(input_size=inplanes, output_size=inplanes, num_experts=6, hidden_size=hidden, k=2, patch_num=patch_num)

        self.ffn1_1 = nn.Linear(hidden, inplanes)


        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        """ New Version """

    def forward(self, x, xi):
        """ Forward pass with input x. """
        B, C, _ = x.shape

        z0 = x[:, 0:int(C/5), :].contiguous()
        x0 = x[:, int(C/5):, :].contiguous()    # x0:RGB  x1:D/T/E

        z1 = xi[:, 0:int(C/5), :].contiguous()
        x1 = xi[:, int(C/5):, :].contiguous()    # x0:RGB  x1:D/T/E



        x_prompted, z_prompted, loss,logits = self.ffn0_1(x1, z1, x0, z0)


        x_prompted = self.ffn1_1(x_prompted) # old version
        z_prompted = self.ffn1_1(z_prompted)

        return x_prompted, z_prompted, loss, logits

def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor, box_mask_z: torch.Tensor):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """
    lens_s = attn.shape[-1] - lens_t    
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return tokens, global_index, None

    attn_t = attn[:, :, :lens_t, lens_t:]

    


    if box_mask_z is not None:
        #print("\n1\n1\n1")
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        # attn_t = attn_t[:, :, box_mask_z, :]
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

        # attn_t = [attn_t[i, :, box_mask_z[i, :], :] for i in range(attn_t.size(0))]
        # attn_t = [attn_t[i].mean(dim=1).mean(dim=0) for i in range(len(attn_t))]
        # attn_t = torch.stack(attn_t, dim=0)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)



    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]
    
    keep_index = global_index.gather(dim=1, index=topk_idx)
    
    removed_index = global_index.gather(dim=1, index=non_topk_idx)
    

    # separate template and search tokens
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_s.shape
    # topk_idx_ = topk_idx.unsqueeze(-1).expand(B, lens_keep, C)

    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    # inattentive_tokens = tokens_s.gather(dim=1, index=non_topk_idx.unsqueeze(-1).expand(B, -1, C))

    # compute the weighted combination of inattentive tokens
    # fused_token = non_topk_attn @ inattentive_tokens
    
    # concatenate these tokens
    # tokens_new = torch.cat([tokens_t, attentive_tokens, fused_token], dim=0)
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    #print("finish ce func")

    return tokens_new, keep_index, removed_index                       # x, global_index_search, removed_index_search


class CEABlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,patch_num=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_MeME = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm2_MeME = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)     #from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

        self.keep_ratio_search = keep_ratio_search



        self.MeME_attn = Prompt_block(inplanes=dim, mlp_rate=4, smooth=True, patch_num=patch_num)


    def forward(self, x, xi, global_index_template, global_index_templatei, global_index_search, global_index_searchi, mask=None, ce_template_mask=None, keep_ratio_search=None):
        
        xori = x
        
        x_attn, attn = self.attn(self.norm1(x), mask, True)

        xi_norm_1 = self.norm1_MeME(xi)
        x_norm_1 = self.norm1_MeME(xori)
        x_feat, z_feat, loss_prompt_attn, logits_attn = self.MeME_attn(x_norm_1, xi_norm_1)
        x_attn_prompted = torch.concat((z_feat, x_feat), dim=1)

        x = x + self.drop_path(x_attn) + self.drop_path(x_attn_prompted)

        # x = x + self.drop_path(x_attn) + self.drop_path(self.adap_t(self.norm1(xi)))                #########-------------------------adapter

        xi_attn, i_attn = self.attn(self.norm1(xi), mask, True)

        xi_feat, zi_feat, _, _ = self.MeME_attn(xi_norm_1, x_norm_1)
        xi_attn_prompted = torch.concat((zi_feat, xi_feat), dim=1)

        xi = xi + self.drop_path(xi_attn) + self.drop_path(xi_attn_prompted)
        # xi = xi + self.drop_path(xi_attn) + self.drop_path(self.adap_t(self.norm1(xori)))           #########-------------------------adapter
                     
        lens_t = global_index_template.shape[1]

        removed_index_search = None
        removed_index_searchi = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
            xi, global_index_searchi, removed_index_searchi = candidate_elimination(i_attn, xi, lens_t, keep_ratio_search, global_index_searchi, ce_template_mask)

        xori = x

        xi_norm_2 = self.norm2_MeME(xi)
        x_norm_2 = self.norm2_MeME(x)

        x_ffn_feat, z_ffn_feat, loss_prompt_ffn, logits_ffn = self.MeME_attn(x_norm_2, xi_norm_2)

        x_ffn_prompted = torch.concat((z_ffn_feat, x_ffn_feat), dim=1)

        x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(x_ffn_prompted)
        # x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adap2_t(xi))   ###-------adapter

        xi_ffn_feat, zi_ffn_feat, _, _ = self.MeME_attn(xi_norm_2, x_norm_2)

        xi_ffn_prompted = torch.concat((xi_ffn_feat, zi_ffn_feat), dim=1)
        xi = xi + self.drop_path(self.mlp(self.norm2(xi))) + self.drop_path(xi_ffn_prompted)
        # xi = xi + self.drop_path(self.mlp(self.norm2(xi))) + self.drop_path(self.adap2_t(self.norm2(xori)))   ###-------adapter
        
        return x, global_index_template, global_index_search, removed_index_search, attn, xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn, loss_prompt_attn+loss_prompt_ffn, logits_attn, logits_ffn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        #print("class Block ")
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        #print("class Block forward")
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
