import sys

from torch.utils.data import DataLoader
from beat_net.beat_net.data_loader import PhysionetECG
import sys


if __name__ == '__main__':
    database_path = sys.argv[1]
    dist_dataloader = DataLoader(dataset=PhysionetECG(database_path=database_path, categories_to_filter=[],
                                                      normalized=False),
                                 batch_size=16,
                                 shuffle=True,
                                 num_workers=10)

    for batch in dist_dataloader:
        print(batch.shape)