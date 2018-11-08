# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import os
import sys
import time
import subprocess
import argparse

import torch
import torch.distributed as dist
from torch.autograd import Variable

def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= num_gpus
    return rt

def init_distributed(rank, num_gpus, group_name, dist_backend, dist_url):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(dist_backend, init_method=dist_url,
                            world_size=num_gpus, rank=rank,
                            group_name=group_name)

def _flatten_dense_tensors(tensors):
    """Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A contiguous 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].contiguous().view(-1)
    flat = torch.cat([t.contiguous().view(-1) for t in tensors], dim=0)
    return flat

def _unflatten_dense_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)

def apply_gradient_allreduce(module):
    """
    Modifies existing model to do gradient allreduce, but doesn't change class
    so you don't need "module"
    """
    if not hasattr(dist, '_backend'):
        module.warn_on_half = True
    else:
        module.warn_on_half = True if dist._backend == dist.dist_backend.GLOO else False

    for p in module.state_dict().values():
        if not torch.is_tensor(p):
            continue
        dist.broadcast(p, 0)

    def allreduce_params():
        if(module.needs_reduction):
            module.needs_reduction = False
            buckets = {}
            for param in module.parameters():
                if param.requires_grad and param.grad is not None:
                    tp = type(param.data)
                    if tp not in buckets:
                        buckets[tp] = []
                    buckets[tp].append(param)
            if module.warn_on_half:
                if torch.cuda.HalfTensor in buckets:
                    print("WARNING: gloo dist backend for half parameters may be extremely slow." +
                          " It is recommended to use the NCCL backend in this case. This currently requires" +
                          "PyTorch built from top of tree master.")
                    module.warn_on_half = False

            for tp in buckets:
                bucket = buckets[tp]
                grads = [param.grad.data for param in bucket]
                coalesced = _flatten_dense_tensors(grads)
                dist.all_reduce(coalesced)
                coalesced /= dist.get_world_size()
                for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                    buf.copy_(synced)

    for param in list(module.parameters()):
        def allreduce_hook(*unused):
            Variable._execution_engine.queue_callback(allreduce_params)
        if param.requires_grad:
            param.register_hook(allreduce_hook)
            dir(param)

    def set_needs_reduction(self, input, output):
        self.needs_reduction = True

    module.register_forward_hook(set_needs_reduction)
    return module


def main(config, stdout_dir, args_str):
    args_list = ['train.py']
    args_list += args_str.split(' ') if len(args_str) > 0 else []

    args_list.append('--config={}'.format(config))

    num_gpus = torch.cuda.device_count()
    args_list.append('--num_gpus={}'.format(num_gpus))
    args_list.append("--group_name=group_{}".format(time.strftime("%Y_%m_%d-%H%M%S")))

    if not os.path.isdir(stdout_dir):
        os.makedirs(stdout_dir)
        os.chmod(stdout_dir, 0o775)

    workers = []

    for i in range(num_gpus):
        args_list[-2] = '--rank={}'.format(i)
        stdout = None if i == 0 else open(
            os.path.join(stdout_dir, "GPU_{}.log".format(i)), "w")
        print(args_list)
        p = subprocess.Popen([str(sys.executable)]+args_list, stdout=stdout)
        workers.append(p)

    for p in workers:
        p.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='JSON file for configuration')
    parser.add_argument('-s', '--stdout_dir', type=str, default=".",
                        help='directory to save stoud logs')
    parser.add_argument(
        '-a', '--args_str', type=str, default='',
        help='double quoted string with space separated key value pairs')

    args = parser.parse_args()
    main(args.config, args.stdout_dir, args.args_str)
