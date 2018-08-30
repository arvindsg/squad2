import torch
from allennlp.modules.time_distributed import TimeDistributed
verbose=False
def ifVerbosePrint(*args):
    if verbose:
        print(*args)

def getAllSubSpans(features,feature_lengths,max_span_length=-1,padToken=torch.zeros([1])):
    passage=features
    passage_lengths=feature_lengths
    max_passage_length=torch.max(passage_lengths).item()

    #i T
    i=torch.arange(end=max_passage_length).unsqueeze(0).expand(max_passage_length,-1)
    j=torch.arange(end=max_passage_length).unsqueeze(1).expand(-1,max_passage_length)
    allIndices=torch.stack([i,j],dim=-1)
    ifVerbosePrint(allIndices,allIndices.shape)
    lengths=j-i+1
    
    
    positiveLengthsMask=(lengths>0)*(lengths<=max_span_length)
    
    
    
    
    ifVerbosePrint(positiveLengthsMask,positiveLengthsMask.shape)
    padTokenIndiceBelowMin=-1
    max_passage_length=passage.size(1)
    padTokenIndice=torch.zeros([1]).long()+max_passage_length
    
    
    validIndices=allIndices.masked_select(positiveLengthsMask.unsqueeze(-1).expand(*positiveLengthsMask.size(),-1))
    
    
#     del i,j,allIndices,positiveLengthsMask
                                          
    validIndices=validIndices.view(-1,2)
    ifVerbosePrint(validIndices,validIndices.shape)
    
    max_length=min(torch.max(lengths),max_span_length)
    expanded_indices_mask=torch.arange(end=max_length)
    
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
    
    answer_lengths=torch.sum(expanded_indices_mask!=padTokenIndiceBelowMin,dim=-1)
    ifVerbosePrint(answer_lengths)
#     raise Exception
    expanded_indices_mask=torch.where(expanded_indices_mask==-1,padTokenIndice,expanded_indices_mask)
    ifVerbosePrint(expanded_indices_mask,expanded_indices_mask.shape)
    # raise Exception
    
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
    
    ifVerbosePrint(features.narrow(dim=-1,start =0,length=1))
    ifVerbosePrint(features.shape,answer_lengths)
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
    
def masked_softmax(vec, mask, dim=1):
    masked_vec = vec * mask.float()
    max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
    exps = torch.exp(masked_vec-max_vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    zeros=(masked_sums == 0)
    masked_sums += zeros.float()
    return masked_exps/masked_sums