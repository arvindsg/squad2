import torch
from utils import getAllSubSpans,getSpanStarts,getSpanEnds
from utils import getIndiceForGoldSubSpan,pointedSelect
from utils import get_best_answers_mask_over_passage,masked_softmax
import torch.nn.modules.conv

#lengths=([10])
passage=torch.LongTensor([[1,2,3,4,5,6,7,8],[3,2,1,0,0,0,0,0],[8,0,0,0,0,0,0,0]]).unsqueeze(-1).float()
passage_lengths=torch.LongTensor([8,3,1])
span_starts=torch.LongTensor([4,2,0]).unsqueeze(-1)
span_ends=torch.LongTensor([6,2,0]).unsqueeze(-1)
print (passage_lengths,passage_lengths.shape)
# raise Exception
padToken=torch.zeros([1])
features,lengths,mask=getAllSubSpans(passage, passage_lengths,5, padToken)
    
print(passage.shape,features.shape,features,mask)
# print()
#out B *T^2 *2
best_span_answer_probs_indice    =getIndiceForGoldSubSpan(span_starts, span_ends, mask)

mask=torch.ByteTensor([[1,1,1,0,0,0,0,0],[1,1,1,0,0,0,0,0],[1,1,1,0,0,0,0,0]])
print(masked_softmax(passage,mask))



best_answers_mask=get_best_answers_mask_over_passage(mask, best_span_answer_probs_indice)
best_span_start=getSpanStarts(best_answers_mask)
best_span_end=getSpanEnds(best_answers_mask)
best_span=torch.stack([best_span_start,best_span_end],dim=-1)
print(best_span,best_span.shape)
print(passage.shape,best_span.shape)
p=pointedSelect(best_span.unsqueeze(1),passage,padToken)

print(p[0],p[0].shape)


