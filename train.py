import os
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


class Train:
    def __init__(self, net, loss, optimizer, jit_script: bool = False):
        self.net = self._init_net(net, jit_script)
        self.loss = loss
        self.optimizer = optimizer

    def _try_all_gpus(self):
        """返回所有可用的GPU，如果没有GPU，则返回[]"""
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return None
        return [torch.device(f'cuda:{i}') for i in range(device_count)]

    def _init_net(self, net, jit_script: bool = False):
        net.apply(self.init_weights)
        if jit_script:
            net = torch.jit.script(net)

        if not torch.cuda.is_available():  # Train with cpu
            self.device = "cpu"
            return net

        if "RANK" not in os.environ or torch.cuda.device_count() <= 1:  # Train with cuda:0
            self.device = torch.device("cuda:0")
            net.to(self.device)
            return net

        # NCCL + DDP
        rank = int(os.environ['RANK'])  # 全局(多机)进程编号
        world_size = int(os.environ['WORLD_SIZE'])  # 全局(多机)总进程数

        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        self.device = torch.device(f'cuda:{rank}')

        net.to(self.device)
        net = DDP(net, device_ids=[rank], find_unused_parameters=False)
        return net

    def grad_clipping(self, theta):
        if isinstance(self.net, nn.Module):
            params = [p for p in self.net.parameters() if p.requires_grad]
        else:
            params = self.net.params

        norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm

    def init_weights(self, m):
        return

    def train_epochs(self, *args, **kwargs):
        return


if __name__ == "__main__":
    print(torch.cuda.device_count())
