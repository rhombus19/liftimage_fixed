import xformers
import xformers.ops
import torch

# q = torch.zeros(([65536, 16, 80])).cuda()
# k = torch.zeros(([65536, 16, 80])).cuda()
# v = torch.zeros(([65536, 16, 80])).cuda()
q = torch.zeros(([65535, 16, 80])).cuda()
k = torch.zeros(([65535, 16, 80])).cuda()
v = torch.zeros(([65535, 16, 80])).cuda()

out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)