import torch
from allennlp.modules.time_distributed import TimeDistributed
verbose=False
def ifVerbosePrint(*args):
    if verbose:
        print(*args)

def getValidSubSpansMask(sequence,sequence_lengths,max_span_length=-1):
    passage=sequence
    passage_lengths=sequence_lengths
    max_passage_length=torch.max(passage_lengths).item()
    assert max_passage_length==passage.size(1),"Max Passage length doesn't align with tensor shapes"
    #i T
    i=torch.arange(end=max_passage_length,device=sequence.device).unsqueeze(0).expand(max_passage_length,-1)
    j=torch.arange(end=max_passage_length,device=sequence.device).unsqueeze(1).expand(-1,max_passage_length)
    allIndices=torch.stack([i,j],dim=-1)
    ifVerbosePrint(allIndices,allIndices.shape)
    lengths=j-i+1
    
    
    positiveLengthsMask=(lengths>0)*(lengths<=max_span_length)
    
    
    
    
    ifVerbosePrint(positiveLengthsMask,positiveLengthsMask.shape)
    padTokenIndiceBelowMin=-1
    
    
    
    
    validIndices=allIndices.masked_select(positiveLengthsMask.unsqueeze(-1).expand(*positiveLengthsMask.size(),-1))
    
    
#     del i,j,allIndices,positiveLengthsMask
                                          
    validIndices=validIndices.view(-1,2)
    ifVerbosePrint(validIndices,validIndices.shape)
    
    max_length=min(torch.max(lengths),max_span_length)
    expanded_indices_mask=torch.arange(end=max_length,device=sequence.device)
    
    expanded_indices_mask=expanded_indices_mask.view(1,*expanded_indices_mask.size()).expand(validIndices.shape[0],*expanded_indices_mask.size())
    ifVerbosePrint(expanded_indices_mask,expanded_indices_mask.shape)
    
    # eim B* 
    #mask
    # validIndicesSpans=(validIndices.narrow(dim=-1,start=0,length=1),validIndices.narrow(dim=-1,start=1,length=1))
    
    # expanded_indices_mask=torch.where(expanded_indices_mask<=validIndicesLengths,ex)
    
    ifVerbosePrint (validIndices,validIndices.shape)
    # raise Exception
    startIndicesExpanded=validIndices.narrow(dim=-1,start=0,length=1).expand(*expanded_indices_mask.size())
    ifVerbosePrint(startIndicesExpanded,startIndicesExpanded.shape)
    
    
    endIndicesExpanded=validIndices.narrow(dim=-1,start=1,length=1).expand(*expanded_indices_mask.size())
    ifVerbosePrint(endIndicesExpanded,endIndicesExpanded.shape)
    # raise Exception
    
    expanded_indices_mask=expanded_indices_mask+startIndicesExpanded
    
    ifVerbosePrint(expanded_indices_mask,expanded_indices_mask.shape)
    
    # raise Exception
    validIndicesMask=(expanded_indices_mask>=startIndicesExpanded).long()*(expanded_indices_mask<=endIndicesExpanded).long()
    
    # validIndicesMask=validIndicesMask.unsqueeze(0).expand(passage_lengths.size(0),*validIndicesMask.size())
    ifVerbosePrint (validIndicesMask,validIndicesMask.shape)
    
    
    # raise Exception
    
    ifVerbosePrint(passage.shape)
    expanded_indices_mask=validIndicesMask*expanded_indices_mask.long()+(validIndicesMask-1)*padTokenIndiceBelowMin*-1
    ifVerbosePrint(expanded_indices_mask,expanded_indices_mask.shape)
    # raise Exception
    
    
    ifVerbosePrint(passage.shape)
    expanded_indices_mask=expanded_indices_mask.unsqueeze(0).expand(passage.size(0),*expanded_indices_mask.shape)
    ifVerbosePrint(expanded_indices_mask,expanded_indices_mask.shape)
    # raise Exception
    
    validIndicesMask=torch.max(expanded_indices_mask,dim=-1)[0]<passage_lengths.view(*passage_lengths.size(),1).expand(*passage_lengths.size(),expanded_indices_mask.size(-2))
    validIndicesMask=validIndicesMask.unsqueeze(-1).expand(*expanded_indices_mask.shape)
    expanded_indices_mask=torch.where(validIndicesMask,expanded_indices_mask,torch.zeros([1]).long()+padTokenIndiceBelowMin)
    
    
#     raise Exception
    ifVerbosePrint(expanded_indices_mask,expanded_indices_mask.shape)
    # raise Exception
    final_mask=expanded_indices_mask
    return final_mask
def getAllSubSpans(features,feature_lengths,max_span_length=-1,padToken=torch.zeros([1])):
    final_mask=getValidSubSpansMask(features, feature_lengths, max_span_length)
    return (*pointedSelect(final_mask, features,padToken),final_mask)
def getSpanEnds(mask):
    return torch.max(mask,dim=-1)[0]
def getSpanStarts(mask):
    maxSpan=torch.max(mask)
    maskLong=mask.long()
    return torch.min(((maskLong!=-1).long()*maskLong)+((maskLong==-1).long()*(maxSpan+1)),dim=-1)[0]
def getIndiceForGoldSubSpan(goldStarts,goldEnds,mask):
    
#     print(mask,mask.shape)
    spanEnds=getSpanEnds(mask)

    spanStarts=getSpanStarts(mask)
    #predictedSpans B*Max_Answers*T
    predictedSpans=torch.stack([spanStarts,spanEnds],dim=-1)
    #goldSpans B*2
    goldSpans=torch.stack([goldStarts,goldEnds],dim=-1)
#     print(predictedSpans)
#     print(goldSpans)
    goldMask=torch.sum(predictedSpans==goldSpans,dim=-1)==2
    goldIndices=torch.argmax(goldMask, dim=1)
    return goldIndices

def get_best_answers_mask_over_passage(answer_features_over_passage_mask,best_answer_indices):   
        return torch.gather(answer_features_over_passage_mask,dim=1,index=best_answer_indices.view(-1,1,1).expand(-1,1,answer_features_over_passage_mask.size(-1))).squeeze(1)


def pointedSelect(expanded_indices_mask,passage,padToken,padTokenIndice=-1):
    max_passage_length=passage.size(1)
    padTokenIndiceDummy=torch.zeros([1]).long()+max_passage_length
    answer_lengths=torch.sum(expanded_indices_mask!=padTokenIndice,dim=-1)
    ifVerbosePrint(answer_lengths)
    expanded_indices_mask=torch.where(expanded_indices_mask==padTokenIndice,padTokenIndiceDummy,expanded_indices_mask)
    
    expanded_indices_mask=expanded_indices_mask.unsqueeze(-1).expand(*expanded_indices_mask.shape,passage.size(-1))
    ifVerbosePrint(expanded_indices_mask,expanded_indices_mask.shape)
    # raise Exception
    
    
#     del validIndicesMask,validIndices
    passage=passage.unsqueeze(1).expand(passage.size(0),expanded_indices_mask.size(1),*passage.shape[1:])
    ifVerbosePrint(passage,passage.shape)
    ifVerbosePrint(padToken.view(1,1,1,1).expand(*passage.size()[:-2],-1,passage.size(-1)).shape)
    passage=torch.cat((passage,padToken.view(1,1,1,1).expand(*passage.size()[:-2],-1,passage.size(-1))),dim=-2)
    ifVerbosePrint (passage.shape)
    features=torch.gather(passage,dim=-2,index=expanded_indices_mask)
    return features,answer_lengths

class BetterTimeDistributed(torch.nn.Module):
    def __init__(self, module):
        super(BetterTimeDistributed, self).__init__()
        self._module = module
        module_wrapped=lambda *inputs:module(list(map(lambda input:input.squeeze(),inputs)))
        self.timeDistributed=TimeDistributed(module_wrapped)
    def forward(self,*inputs):
        wrapped_inputs=list(map(lambda input:input.unsqueeze(-1),inputs))
        return self.timeDistributed(wrapped_inputs)
    
def masked_softmax(vec, mask, dim=-1):
    masked_vec = vec * mask.float()+((mask==0).float()*-1E7)
    return torch.nn.functional.softmax(masked_vec,dim)

