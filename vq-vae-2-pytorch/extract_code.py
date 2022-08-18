import argparse
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import lmdb
from tqdm import tqdm

from dataset import ImageFileDataset, CodeRow
from vqvae import VQVAE


def extract(lmdb_env, loader, model, device):
    index = 0

    with lmdb_env.begin(write=True) as txn:
        pbar = tqdm(loader)

        for img, label in pbar:
            img = img.to(device)

            _, _, _, id_t, id_b = model.encode(img)
            id_t  = id_t.detach().cpu().numpy()
            id_b  = id_b + 512
            id_b  = id_b.detach().cpu().numpy()
            label = label.unsqueeze(1).detach().cpu().numpy()

            for top, bottom, label in zip(id_t, id_b, label):
                row = CodeRow(top=top, bottom=bottom, label=label)
                txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                index += 1
                pbar.set_description(f'inserted: {index}')

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--ckpt', type=str, default='/home/ubuntu/VQVAE/vq-vae-2-pytorch/checkpoint/vqvae_070.pt')
    parser.add_argument('--name', type=str, default= 'exp1')
    # parser.add_argument('path', type=str)

    args = parser.parse_args()

    device = 'cuda'

    transform = transforms.Compose(
        [
            # transforms.Resize(args.size),
            # transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = datasets.CIFAR10(root= '/home/ubuntu/exp_temp', download= True, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    model = VQVAE()
    model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    model.eval()

    map_size = 100 * 1024 * 1024 * 1024

    env = lmdb.open(args.name, map_size=map_size)

    extract(env, loader, model, device)
