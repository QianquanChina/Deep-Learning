import torch
import numpy as np
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.utils.checkpoint as checkpoint

def drop_path_f(x, drop_prob : float = 0., training : bool = False):# {{{

    if drop_prob == 0. or not training:

        return x

    keep_prob = 1 - drop_prob
    shape = ( x.shape[0], ) + ( 1, ) * ( x.ndim - 1 )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand( shape, dtype = x.dtype, device = x.device )
    random_tensor.floor_()  # binarize
    output = x.div( keep_prob ) * random_tensor

    return output# }}}

class DropPath(nn.Module):# {{{

    def __init__(self, drop_prob=None):

        super(DropPath, self).__init__()

        self.drop_prob = drop_prob

    def forward(self, x):

        return drop_path_f( x, self.drop_prob, self.training )# }}}

def window_partition(x, window_size : int):# {{{
    """
        将feature map 划分成一个个没有重叠的windows

        Args:
            x: ( B, H, W, C )
            window_size (int): window size

        Returns:
            windows: ( num_windows*B, window_size, window_size, C )
    """
    B, H, W, C = x.shape

    # view : x = [ B, H, W, C ] -> [ B, H//M, M, W//M, M, C ] 
    x          = x.view( B, H // window_size, window_size, W // window_size, window_size, C )

    # perm : x = [ B, H//M, M, W//M, M, C ] -> [ B, H//M, W//M, M, M,  C ]
    # view : x = [ B, H//M, W//M, M, M, C ] -> [ B*num_windows, Mh, Mw C ]
    windows    = x.permute( 0, 1, 3, 2, 4, 5 ).contiguous().view( -1, window_size, window_size, C )

    return windows# }}}

def window_reverse(windows, window_size, H, W):# {{{

    """
        将窗口变成原来的feature map
        
        Args:
            windows: ( num_windows*B, window_size, window_size, C )
            window_size (int): Window size
            H (int): Height of image
            W (int): Width of image

        Returns:
             x: (B, H, W, C)
    """
    B = int( windows.shape[0] / ( H * W / window_size / window_size ) )

    # view : x = [ B*num_windows, M, M, C ] -> [ B, H//M, W//M, M, M, C ] 
    x = windows.view( B, H // window_size, W // window_size, window_size, window_size, -1 )

    # perm : x = [ B, H//M, W//M, M, M, C ] -> [ B, H//M, M, W//M, M, C ]
    # view : x = [ B, H//M, M, W//M, M, C ] -> [ B, H, W, C ] 
    x = x.permute( 0, 1, 3, 2, 4, 5 ).contiguous().view( B, H, W, -1 )

    return x# }}}

class PatchEmbedb(nn.Module):# {{{

    def __init__(self, patch_size = 4, in_c = 3, embed_dim = 96, norm_layer = None):

        super().__init__()
        patch_size      = ( patch_size, patch_size )
        self.patch_size = patch_size
        self.in_chans   = in_c
        self.embed_dim  = embed_dim
        self.proj       = nn.Conv2d( in_c, embed_dim, kernel_size = patch_size, stride = patch_size )
        self.norm       = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):

        _, _, H, W, = x.shape

        # 如果输入的图片H，W不是patch_size的整数倍就要padding
        if H % self.patch_size[0] != 0:
            
            #( image , w_left, w_right, h_top, h_bottom, c_front, c_back )
            x  = F.pad( x, ( 0, 0, 0, self.patch_size[0] - H % self.patch_size[0] ) )

        if W % self.patch_size[1] != 0:

            x  = F.pad( x, ( 0, self.patch_size[1] - W % self.patch_size[1] ) )

 
        # 下采样处理 
        x = self.proj(x)

        _, _, H, W, = x.shape

        # 展平 
        # flatten   : [ B, C, H, W ] -> [ B, C, HW ]
        # transpose : [ B, C, HW   ] -> [ B HW, C  ]
        x = x.flatten(2).transpose( 1, 2 )
        x = self.norm(x)

        return x, H, W# }}}

class PatchMerging(nn.Module):# {{{

    """
        Args:
            dim        : feature map 的通道.
            norm_layer : 归一化方法.

    """

    def __init__(self, dim, norm_layer = nn.LayerNorm):

        super().__init__()

        self.dim       = dim
        self.reduction = nn.Linear( 4 * dim, 2 * dim, bias = False )
        self.norm      = norm_layer( 4 * dim )

    def forward(self, x, H, W):
        
        """
            x : [ B, HW, C ]

        """
        B, L, C = x.shape

        assert L == H * W, 'input feature has wrong size'
         
        # 恢复原来的样子
        x = x.view( B, H, W, C )

        # padding
        # 如果输入的feater map 的 H，W不是2的整数倍就要padding
        pad_input = ( H % 2 == 1 ) or ( W % 2 == 1 )

        if pad_input:
            
            x = F.pad( x, ( 0, 0, 0, W % 2, 0, H % 2) )


        x0 = x[ :, 0::2, 0::2, : ] # [ B, H/2, W/2, C ] 
        x1 = x[ :, 1::2, 0::2, : ] # [ B, H/2, W/2, C ]  
        x2 = x[ :, 0::2, 1::2, : ] # [ B, H/2, W/2, C ]  
        x3 = x[ :, 1::2, 1::2, : ] # [ B, H/2, W/2, C ]  

        
        x  = torch.cat( [ x0, x1, x2, x3 ], -1 ) # [ B, H/2, W/2, 4*C ] 

        # x : [ B, H/2, W/2, 4*C ] -> [ B, H/2*W/2, 4*C ] 
        x  = x.view( B, -1, 4 * C )
        x  = self.norm(x)

        # x : [ B, H/2*W/2, 4*C ] -> [ B, H/2*W/2, 2*C ] 
        x  = self.reduction(x)

        return x# }}}

class Mlp(nn.Module):# {{{

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):

        super().__init__()

        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1   = nn.Linear( in_features, hidden_features )
        self.act   = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2   = nn.Linear( hidden_features, out_features )
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x# }}}

class WindowAttention(nn.Module):# {{{

    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

       Args:

            dim (int): Number of input channels.
            window_size (tuple[int]): The height and width of the window.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
            attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
            proj_drop (float, optional): Dropout ratio of output. Default: 0.0

    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()

        self.dim         = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = head_dim ** -0.5

        # 创建相对置偏移表
        self.relative_position_bias_table = Parameter(

                                                         torch.zeros(
                                                                        ( 2 * window_size[0] - 1 ) * ( 2 * window_size[1] - 1 ),
                                                                        num_heads
                                                                    )

                                                      )  # [ 2*Wh-1 * 2*Ww-1, nH ]

        # 生成相对位置坐标以及相对位置的索引
        coords_h = torch.arange( self.window_size[0] )
        coords_w = torch.arange( self.window_size[1] )
        coords   = torch.stack( torch.meshgrid( [ coords_h, coords_w ], indexing = "ij" ) )  # [ 2, Wh, Ww ]

        coords_flatten  = torch.flatten( coords, 1 )                        # [ 2, Wh*Ww ]

        # 获得相对位置索引
        relative_coords = coords_flatten[ :, :, None ] - coords_flatten[ :, None, : ]  # [ 2, Wh*Ww, Wh*Ww ]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()                # [ Wh*Ww, Wh*Ww, 2 ]

        # 将二元索引变成一元索引
        relative_coords[ :, :, 0 ] += self.window_size[0] - 1                          # shift to start from 0
        relative_coords[ :, :, 1 ] += self.window_size[1] - 1
        relative_coords[ :, :, 0 ] *= 2 * self.window_size[1] - 1

        # -----------
        relative_position_index = relative_coords.sum(-1)                              # [ Wh*Ww, Wh*Ww ]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv       = nn.Linear( dim, dim * 3, bias=qkv_bias )
        self.attn_drop = nn.Dropout( attn_drop )
        self.proj      = nn.Linear( dim, dim )
        self.proj_drop = nn.Dropout( proj_drop )

        nn.init.trunc_normal_( self.relative_position_bias_table, std=.02 ) 
        self.softmax = nn.Softmax( dim=-1 )

    def forward(self, x, mask : Optional[torch.Tensor] = None):

        """ Forward function.

            Args:

                x: input features with shape of (num_windows*B, N, C)
                mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None

        """

        B_, N, C = x.shape
        
        # qkv():   -> [ batch_size*num_windows, Mh*Mw, 3*total_embed_dim ]
        # reshape: -> [ batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head ]
        # permute: -> [ 3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head ]

        qkv     = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q    = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[  
                                                                      self.relative_position_index.view(-1) # type:ignore

                                                                  ].view(
                                                                            self.window_size[0] * self.window_size[1],
                                                                            self.window_size[0] * self.window_size[1], 
                                                                            -1 
                                                                        )  # Wh*Ww,Wh*Ww,nH

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:

            # mask 矩阵与注意力相加
            nW   = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)

        else:

            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = ( attn @ v ).transpose( 1, 2 ).reshape( B_, N, C )

        # 多头融合
        x = self.proj(x)
        x = self.proj_drop(x)

        return x# }}}

class SwinTransformerBlock(nn.Module):# {{{

    def __init__( 
                    self,
                    dim,
                    num_heads,
                    window_size = 7,
                    shift_size  = 0,
                    mlp_ratio   = 4.,
                    qkv_bias    = True,
                    drop        = 0.,
                    attn_drop   = 0.,
                    drop_path   = 0.,
                    act_layer   = nn.GELU,
                    norm_layer  = nn.LayerNorm
                ):

        super().__init__()
        
        self.dim         = dim
        self.num_heads   = num_heads
        self.window_size = window_size
        self.shift_size  = shift_size
        self.mlp_ratio   = mlp_ratio

        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)
        self.attn  = WindowAttention(

                                        dim,
                                        window_size = ( self.window_size, self.window_size ),
                                        num_heads   = num_heads,
                                        qkv_bias    = qkv_bias,
                                        attn_drop   = attn_drop,
                                        proj_drop   = drop

                                    )

        self.drop_path = DropPath( drop_path ) if drop_path > 0. else nn.Identity()
        self.norm2     = norm_layer(dim)
        mlp_hidden_dim = int( dim * mlp_ratio )
        self.mlp       = Mlp( in_features = dim, hidden_features = mlp_hidden_dim, act_layer = act_layer, drop = drop )

    def forward(self, x, attn_mask):

        """ Forward function.
           Args:
                x: Input feature, tensor size (B, H*W, C).
                H, W: Spatial resolution of the input feature.
                attn_mask: Attention mask for cyclic shift.
        """
        B, L, C  = x.shape
        H, W     = self.H, self.W

        assert L == H * W, 'input feature has wrong size' # type: ignore

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = ( self.window_size - W % self.window_size ) % self.window_size
        pad_b = ( self.window_size - H % self.window_size ) % self.window_size
        x = F.pad( x, ( 0, 0, pad_l, pad_r, pad_t, pad_b ) )
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:

            # 移动windows到相应的位置
            shifted_x = torch.roll( x, shifts = ( -self.shift_size, -self.shift_size ), dims = ( 1, 2 ) )

        else:

           shifted_x = x
           attn_mask = None

        # partition windows
        x_windows = window_partition( shifted_x, self.window_size )             # [ nW*B, window_size, window_size, C ]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [ nW*B, window_size*window_size , C ]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [ nW*B, window_size*window_size, C ]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C) # [ nW*B, Mh, Mw, C ]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)          # [ B H' W' C ]

        # reverse cyclic shift
        if self.shift_size > 0:

            x = torch.roll( shifted_x, shifts = ( self.shift_size, self.shift_size ), dims = ( 1, 2 ) )

        else:

            x = shifted_x

        if pad_r > 0 or pad_b > 0:

            # 如果前面有pad那么将其移除掉
            x = x[ :, :H, :W, : ].contiguous()

        x = x.view( B, H * W, C ) # type: ignore

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path( self.mlp( self.norm2(x) ) )

        return x# }}}

class BasicLayer(nn.Module):# {{{

    def __init__( 
                    self,
                    dim , 
                    depth,
                    num_heads, 
                    window_size,
                    mlp_ratio      = 4.,
                    qkv_bias       = True,
                    drop           = 0.,
                    attn_drop      = 0.,
                    drop_path      = 0.,
                    norm_layer     = nn.LayerNorm,
                    downsample     = None,
                    use_checkpoint = False
                ):
        
        super().__init__()

        self.dim            = dim 
        self.depth          = depth
        self.window_size    = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size     = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList(
                                       [
                                           SwinTransformerBlock(
                                                                   dim         = dim,
                                                                   num_heads   = num_heads,
                                                                   window_size = window_size,
                                                                   shift_size  = 0 if ( i % 2 == 0 ) else self.shift_size,
                                                                   mlp_ratio   = mlp_ratio,
                                                                   qkv_bias    = qkv_bias,
                                                                   drop        = drop,
                                                                   attn_drop   = attn_drop,
                                                                   drop_path   = drop_path[i] if isinstance( drop_path, list) else drop_path,
                                                                   norm_layer  = norm_layer

                                                               )

                                           for i in range(depth)

                                       ]    

                                   )

        # patch merging layer
        if downsample is not None:

            self.downsample = downsample( dim = dim, norm_layer = norm_layer )

        else:

            self.downsample = None

    def create_mask(self, x, H, W):

        """
                计算SW-MSA的掩码

        """
        # 保证Hp Wp是windows_size的整数倍
        Hp = int( np.ceil( H / self.window_size ) ) * self.window_size
        Wp = int( np.ceil( W / self.window_size ) ) * self.window_size

        # 拥有和feature map 一样的通道顺序，方便后续的windows_partition
        img_mask = torch.zeros( (1, Hp, Wp, 1 ), device = x.device )  # 1 Hp Wp 1

        h_slices = ( 
                       slice( 0, -self.window_size ),
                       slice( -self.window_size, -self.shift_size ),
                       slice( -self.shift_size, None )
                   )
                       
        w_slices = (
                       slice( 0, -self.window_size ),
                       slice( -self.window_size, -self.shift_size ),
                       slice( -self.shift_size, None )
                   )
        cnt = 0

        for h in h_slices:

            for w in w_slices:

                img_mask[ :, h, w, : ] = cnt

                cnt += 1

        # 将img_mask分成所需要的window 因为img_mask的第一维度是1所以nW就是window的个数.
        mask_windows = window_partition( img_mask, self.window_size )  # [ nW, window_size, window_size, 1 ]
        mask_windows = mask_windows.view( -1, self.window_size * self.window_size ) #[ nW, Mh*Mw ]
        attn_mask    = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask    = attn_mask.masked_fill( attn_mask != 0, float(-100.0) ).masked_fill( attn_mask == 0, float(0.0) )

        return attn_mask

    def forward(self, x, H, W):

        """
            args:
                x : [ B, HW, C ].

        """
        attn_mask = self.create_mask( x, H, W )

        for blk in self.blocks:
            
            # 给当前的block添加H W的属性.
            blk.H, blk.W = H, W 

            if self.use_checkpoint:

                x = checkpoint.checkpoint( blk, x, attn_mask )
            
            else:

                x = blk( x, attn_mask )

        if self.downsample is not None:

            x    = self.downsample( x, H, W )
            H, W = ( H + 1 ) //2, ( W + 1 ) // 2

        return x, H, W# }}}

class SwinTransformer(nn.Module):# {{{

    """
        Args:
            patch_size     : 下采样的倍数.
            in_chans       : 输入图片的通道.
            num_classses   : 分类的个数.
            depths         : 每层stage中SwinTransformerBlock的重复次数.
            num_head       : Mutil Head Self Attention中head的个数.
            window_size    : Window Mutil Head Self Attention 中窗口的大小.
            mlp_ratio     : SwinTransformerBlock中的MLP层中第一个全连接层将我们的channel翻的倍数. 
            qkv_bias       : 是否使用qkv的相对位置偏置.
            drop_rate      :
            att_drop_rate  : 在Mutil Head Self Attention 中所采用.
            drop_path_rate : 在SwinTransformer中所用的, 这是从0递增到我们所设定的数值.
            norm_layer     : 归一化.
            patch_norm     : 
            use_checkpoint : 默认不使用,使用的话会节省内存.

    """

    def __init__(
                    self,
                    patch_size  = 4           , in_chans       = 3             , num_classes    = 1000             ,
                    embed_dim   = 96          , depths         = ( 2, 2, 6, 2 ), num_heads       = ( 3, 6, 12, 24 ),
                    window_size = 7           , mlp_ratio      = 4.            , qkv_bias        = True            , 
                    drop_rate   = 0.          , attn_drop_rate = 0.            , drop_path_rate  = 0.1             ,
                    norm_layer  = nn.LayerNorm, patch_norm     = True          , use_checkpoint  = False           , 
                    out_indices=(0, 1, 2, 3)  , **kwargs
                ):

        super().__init__()
        
        self.num_classes  = num_classes
        self.num_layers   = len(depths)
        self.embed_dim    = embed_dim
        self.patch_norm   = patch_norm
        self.out_indices  = out_indices

        # stage4 输出的特征矩阵的channels
        self.num_features = [ int( embed_dim * 2 ** i ) for i in range( self.num_layers ) ]
        self.mlp_ratio    = mlp_ratio
         
        # 将图片划分为没有重叠的patch.
        self.patch_embed  = PatchEmbedb(
                                          patch_size = patch_size, 
                                          in_c       = in_chans, 
                                          embed_dim  = embed_dim,
                                          norm_layer = norm_layer if self.patch_norm else None
                                       )
        
        self.pos_drop = nn.Dropout( p = drop_rate )
        
        # 对不同的TransformerBlock生成不同的dpr
        dpr = [ x.item() for x in torch.linspace( 0, drop_path_rate, sum(depths) ) ]

        self.layers = nn.ModuleList()

        # build layers
        for i_layer in range( self.num_layers ):

            # 这里的stage不包含本层的stage的patch_merging，包含下层的patch_merging
            layers = BasicLayer(
                                  dim            = int( embed_dim * 2 ** i_layer ),
                                  depth          = depths[i_layer],
                                  num_heads      = num_heads[i_layer],
                                  window_size    = window_size,
                                  mlp_ratio      = self.mlp_ratio,
                                  qkv_bias       = qkv_bias,
                                  drop           = drop_rate,
                                  attn_drop      = attn_drop_rate,
                                  drop_path      = dpr[ sum( depths[ : i_layer ] ) : sum( depths[ : i_layer + 1 ] ) ],
                                  norm_layer     = norm_layer,
                                  downsample     = PatchMerging if ( i_layer < self.num_layers - 1 ) else None,
                                  use_checkpoint = use_checkpoint
                               )

            self.layers.append(layers)

        # ----------- 分类任务 -------------
            
        self.norm    = norm_layer( self.num_features[ self.num_layers - 1 ] )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head    = nn.Linear( self.num_features[ self.num_layers - 1 ], num_classes ) if num_classes > 0 else nn.Identity() 

        # ----------------------------------

        self.apply( self._init_weights ) 

    def _init_weights(self, m): 

        if isinstance( m, nn.Linear ):

            nn.init.trunc_normal_( m.weight, std = .02 )

            if isinstance( m, nn.Linear ) and m.bias is not None:

                nn.init.constant_( m.bias, 0 )

        elif isinstance( m, nn.LayerNorm ):

            nn.init.constant_( m.bias, 0 )
            nn.init.constant_( m.weight, 1.0 )

    def forward(self, x):
        
        # x : [ B, C, H, W ] -> [ B, HW, C ]
        x, H, W = self.patch_embed(x) 
        x       = self.pos_drop(x)

        layers_out = []

        for i in range(self.num_layers):

            layer = self.layers[i]
            x, H, W = layer( x, H, W )

            if i in self.out_indices:

                layer_out = x.view( -1, H, W, self.num_features[i] ).permute( 0, 3, 1, 2 ).contiguous()
                layers_out.append(layer_out)
            

        # ----------- 分类任务 -------------

        x = self.norm(x) # x : [ B, HW, C ]

        # transpose : [ B, HW, C ] -> [ B, C, HW ] , avgpool : [ B, C, HW ] -> [ B, C, 1 ]
        x = self.avgpool( x.transpose( 1, 2 ) )
        x = torch.flatten( x, 1 )
        x = self.head(x)

        # ----------------------------------

        return x# }}}

def swin_tiny_patch4_window7_224(num_classes : int = 1000, **kwargs):# {{{

    model = SwinTransformer( 
                               in_chans     = 3, 
                               patch_size   = 4,
                               window_size  = 7,
                               embed_dim    = 96,
                               depths       = ( 2, 2, 6, 2 ),
                               num_heads    = ( 3, 6, 12, 24 ),
                               num_classes = num_classes,
                               **kwargs
                           )

    return model# }}}

