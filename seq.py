from keras.utils import Sequence
import pandas as pd
import numpy as np


class DNASequence(Sequence):
    def __init__(self, csvfile: str, batch_size: int = None):
        self.df = pd.read_csv(csvfile)

        self.batch_size = batch_size if batch_size is not None else len(self.df)

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, item):

        batch_input = np.zeros((self.batch_size, 60, 4))
        batch_output = np.zeros((self.batch_size, 3))

        for bidx in range(0, self.batch_size):

            sidx = self.batch_size*item + bidx

            sample = self.df.iloc[sidx]

            dna_vector_map = {
                'A': [1, 0, 0, 0],
                'G': [0, 1, 0, 0],
                'T': [0, 0, 1, 0],
                'C': [0, 0, 0, 1],
                'D': [1, 1, 1, 0],
                'N': [1, 1, 1, 1],
                'S': [0, 1, 0, 1],
                'R': [1, 1, 0, 0]
            }

            cls_map = {
                'N': [1, 0, 0],
                'EI': [0, 1, 0],
                'IE': [0, 0, 1]
            }

            for didx, d in enumerate(sample['seq']):
                batch_input[bidx, didx] = dna_vector_map[d]

            batch_output[bidx] = cls_map[sample['cls']]

        return batch_input, batch_output


if __name__ == '__main__':
    seq = DNASequence('data/train.csv')

    bin, bout = seq.__getitem__(0)

    for i in range(0, bin.shape[0]):
        print('-----')
        print(bin[i])
        print(bout[i])
        print('-----')
