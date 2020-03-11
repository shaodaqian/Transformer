import torch.nn as nn
import torch
import numpy as np
import os
import torchtext as tt
from torchtext.datasets import TranslationDataset
from torchtext.datasets import WMT14
import spacy



# en-de: Byte-pair encoding: Massive exploration of neural machine translation architectures (2017), https://arxiv.org/pdf/1703.03906.pdf
"""We tokenize and clean all datasets with the
scripts in Moses2
and learn shared subword units
using Byte Pair Encoding (BPE) (Sennrich et al.,
2016b) using 32,000 merge operations for a final
vocabulary size of approximately 37k.
We release our data preprocessing
scripts together with the NMT framework to the
public. For more details on data preprocessing parameters,
we refer the reader to the code release."""
# https://github.com/google/seq2seq
# https://github.com/google/seq2seq/blob/master/docs/nmt.md#download-data



# en-fr: 32000 word-piece vocab: Google’s neural machine translation system: Bridging the gap between human and machine translation (2016)


