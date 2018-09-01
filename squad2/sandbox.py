import torch
from utils import getAllSubSpans
#lengths=([10])
passage=torch.randn(3,8,6)
passage_lengths=torch.LongTensor([8,3,1])
print (passage_lengths,passage_lengths.shape)
# raise Exception
padToken=torch.zeros([1])
features,lengths,mask=getAllSubSpans(passage, passage_lengths,5, padToken)
print(passage.shape,features.shape,features,mask)
#out B *T^2 *2
