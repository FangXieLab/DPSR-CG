

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset



def validation(model, test_loader,device):
    model.eval()
    num_examples = 0
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.cross_entropy(output, target, reduction='sum')

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_examples += len(data)
    test_loss /= num_examples
    test_acc = 100. * correct / num_examples

    return test_loss, test_acc


def validation_vit(model, test_loader, device, real_batch_size=64):
    model.eval()
    num_examples = 0
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            # 此时 data 可能是一个巨大的 Batch (比如 1000)
            # 注意：不要在这里直接 data.to(device)，因为大 Batch 上 GPU 会瞬间 OOM

            # 1. 使用 torch.split 将大 Batch 切分成小的物理 Batch
            # 默认在 CPU 上切分，不占显存
            data_chunks = torch.split(data, real_batch_size)
            target_chunks = torch.split(target, real_batch_size)

            # 2. 循环处理每个小切片
            for small_data, small_target in zip(data_chunks, target_chunks):
                # 只有当前的小切片才移动到 GPU
                small_data = small_data.to(device)
                small_target = small_target.to(device)

                output = model(small_data)

                # 累加 Loss (使用 reduction='sum' 方便累加)
                test_loss += F.cross_entropy(output, small_target, reduction='sum').item()

                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(small_target.view_as(pred)).sum().item()

            # 统计总样本数
            num_examples += len(data)

    # 计算平均值
    test_loss /= num_examples
    test_acc = 100. * correct / num_examples

    return test_loss, test_acc


def validation_per_sample(model, test_loader,device,C):
    model.eval()
    num_examples = 0
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data,target=data.to(device),target.to(device)
            for x,y in TensorDataset(data,target):
                output = model(torch.unsqueeze(x,0))
                if len(output.shape)==2:
                    output=torch.squeeze(output,0)
                loss=F.cross_entropy(output, y, reduction='sum')
                loss=min(loss,C)   #逐样本loss裁剪
                test_loss += loss
            num_examples += len(data)
    test_loss /= num_examples

    # print(f'Test set: Average loss: {test_loss:.4f}, '
    #       f'Accuracy: {correct}/{num_examples} ({test_acc:.2f}%)')

    return test_loss
