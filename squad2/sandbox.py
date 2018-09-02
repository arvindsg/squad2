import torch
from utils import getAllSubSpans
from utils import getIndiceForGoldSubSpan
from squad2.utils import get_best_answers_mask_over_passage
import torch.nn.modules.conv
#lengths=([10])
passage=torch.randn(3,8,6)
passage_lengths=torch.LongTensor([8,3,1])
span_starts=torch.LongTensor([4,1,0])
span_ends=torch.LongTensor([6,2,0])
print (passage_lengths,passage_lengths.shape)
# raise Exception
padToken=torch.zeros([1])
features,lengths,mask=getAllSubSpans(passage, passage_lengths,5, padToken)
    
print(passage.shape,features.shape,features,mask)
getIndiceForGoldSubSpan(span_starts, span_ends, mask)
#out B *T^2 *2
best_span_answer_probs_indice    =torch.IntTensor([ 12,  3,0])




get_best_answers_mask_over_passage(mask, best_span_answer_probs_indice)