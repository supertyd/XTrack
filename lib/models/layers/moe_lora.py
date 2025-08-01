
# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


from functools import partial
from timm.models.layers import to_2tuple
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from torch.autograd import Variable











class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, top_gedits, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(1))
            # stitched = stitched.mul(top_gedits.unsqueeze(1))
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out


class Fusion(nn.Module):
    def  __init__(self, rank, norm_layer=partial(nn.LayerNorm,eps=1e-6), act_layer=nn.GELU,):
        super(Fusion, self).__init__()
        self.norm = norm_layer
        self.act = act_layer()
        self.conv = nn.Conv2d(rank, rank, kernel_size=3, padding=1, bias=False, groups= rank)

        self.proj_xx = nn.Linear(rank, rank)
        self.proj_x = nn.Linear(rank, rank)
        self.proj_mod = nn.Linear(rank, rank)

        self.norm = nn.GELU()

    def forward(self, x, mod):
        shortcut = x
        x = self.norm(x)
        mod = self.norm(mod)

        x = self.proj_x(x)
        x = token2feature(x)
        x = feature2token(x)
        mod = self.proj_mod(mod)
        x = self.act(mod) * x
        x = self.proj_xx(x)
        x = x + shortcut
        return x

def token2feature(tokens):
    B,L,D=tokens.shape
    H=W=int(L**0.5)
    x = tokens.permute(0, 2, 1).view(B, D, W, H).contiguous()
    return x



def feature2token(x):
    B,C,W,H = x.shape
    L = W*H
    tokens = x.view(B, C, L).permute(0, 2, 1).contiguous()
    return tokens

class LaplacianConvNet(nn.Module):
    def __init__(self, rank):
        super(LaplacianConvNet, self).__init__()
        self.rank = rank
        self.conv = nn.Conv2d(self.rank, self.rank, kernel_size=3, padding=1, bias=False, groups= self.rank)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        return x

    def _initialize_weights(self):
        # Define the Laplacian filter
        laplacian_filter = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32)

        # Adjust the shape of the laplacian_filter to match (out_channels, in_channels, kernel_height, kernel_width)
        laplacian_filter = laplacian_filter.view(1, 1, 3, 3)
        laplacian_filter = laplacian_filter.repeat(self.rank, 1, 1, 1)

        # Initialize the weights of the conv layer
        with torch.no_grad():
            self.conv.weight.copy_(laplacian_filter)


class Shared(nn.Module):
    def __init__(self, rank, norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, ):
        super(Shared, self).__init__()
        self.norm = norm_layer(rank)
        self.act = act_layer()
        self.conv = LaplacianConvNet(rank)
        # self.conv = nn.Sequential(nn.Conv2d(rank, rank, kernel_size=(1, 3), padding=(0, 1), bias=False, groups= rank), nn.Conv2d(rank, rank, kernel_size=(3, 1), padding=(1, 0), bias=False, groups= rank))
        #self.conv = nn.Conv2d(rank, rank, kernel_size=3, padding=1, bias=False, groups= rank)
        #self.conv = nn.Sequential(nn.Conv2d(rank, rank, kernel_size=(1, 3), padding=(0, 1), bias=False, groups=rank),
        #                          nn.Conv2d(rank, rank, kernel_size=(3, 1), padding=(1, 0), bias=False, groups=rank))
        self.proj_x = nn.Linear(rank, 2*rank)
        self.proj_xx = nn.Linear(rank, rank)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.proj_x(x)
        x1, x2 = x.chunk(2, dim=2)
        x1 = token2feature(x1)
        x1 = self.conv(x1)
        x1 = feature2token(x1)

        x = self.act(x1) * x2
        x = self.proj_xx(x)
        x = x + shortcut
        return x




class MoE_lora(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, output_size, num_experts, hidden_size, noisy_gating=True, k=2,patch_num=None):
        super(MoE_lora, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        # instantiate experts
        self.experts = nn.ModuleList([nn.Linear(self.input_size, self.hidden_size) for i in range(self.num_experts)])
        self.shared_expert = nn.Linear(self.input_size, self.hidden_size)
        self.shared_expert1 = Shared(self.hidden_size)

        self.dte = nn.Linear(self.hidden_size, self.hidden_size)
        self.dteall = nn.Linear(self.hidden_size, self.hidden_size)

        self.rgb = nn.Linear(self.input_size, self.hidden_size)

        self.fuse = Fusion(self.hidden_size)



        self.w_gate = nn.Parameter(torch.zeros(patch_num, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(patch_num, num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, z, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """

        x_mean = torch.mean(x, dim=-1)
        z_mean = torch.mean(z, dim=-1)
        x_mean  = torch.concat((x_mean,z_mean),dim=-1)

        clean_logits = x_mean @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x_mean @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)  # normalization
        self.logits = top_k_logits


        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load, self.softmax(logits), top_k_logits

    def forward(self, x, z, x0, z0, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """

        gates, load, logits,top_k_logits = self.noisy_top_k_gating(x, z, self.training)
        # calculate importance loss

        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)

        shared_x = self.shared_expert1(self.shared_expert(x))
        shared_z = self.shared_expert1(self.shared_expert(z))

        rgbx = self.rgb(x0)
        rgbz = self.rgb(z0)

        expert_inputs_x = dispatcher.dispatch(x)
        expert_inputs_z = dispatcher.dispatch(z)



        expert_outputs_x = [self.experts[i](expert_inputs_x[i]) for i in range(self.num_experts)]
        expert_outputs_z = [self.experts[i](expert_inputs_z[i]) for i in range(self.num_experts)]




        x = self.dteall(self.dte(dispatcher.combine(expert_outputs_x, top_k_logits)) + shared_x)
        z = self.dteall(self.dte(dispatcher.combine(expert_outputs_z, top_k_logits)) + shared_z)


        # x = self.dteall(shared_x)
        # z = self.dteall(shared_z)

        x = self.fuse(rgbx, x)
        z = self.fuse(rgbz, z)

        return x, z, loss, logits
