import torch
#lengths=([10])
passage=torch.randn(8,6)
max_passage_length=8

#i T
i=torch.arange(end=max_passage_length).unsqueeze(0).expand(max_passage_length,-1)
j=torch.arange(end=max_passage_length).unsqueeze(1).expand(-1,max_passage_length)
allIndices=torch.stack([i,j],dim=-1)
print(allIndices,allIndices.shape)
lengths=j-i+1


positiveLengthsMask=lengths>=0




print(positiveLengthsMask,positiveLengthsMask.shape)
validIndices=allIndices.masked_select(positiveLengthsMask.unsqueeze(-1).expand(*positiveLengthsMask.size(),-1))
                                      
validIndices=validIndices.view(-1,2)
print(validIndices,validIndices.shape)
# list(validIndices.shape)
expanded_indices_mask=torch.arange(end=torch.max(lengths))

expanded_indices_mask=expanded_indices_mask.view(1,*expanded_indices_mask.size()).expand(validIndices.shape[0],*expanded_indices_mask.size())
print(expanded_indices_mask,expanded_indices_mask.shape)
# eim B* 
#mask
# validIndicesSpans=(validIndices.narrow(dim=-1,start=0,length=1),validIndices.narrow(dim=-1,start=1,length=1))

# expanded_indices_mask=torch.where(expanded_indices_mask<=validIndicesLengths,ex)
startIndicesExpanded=validIndices.narrow(dim=-1,start=0,length=1).expand(*expanded_indices_mask.size())
print(startIndicesExpanded,startIndicesExpanded.shape)
# raise Exception

endIndicesExpanded=validIndices.narrow(dim=-1,start=1,length=1).expand(*expanded_indices_mask.size())
expanded_indices_mask=expanded_indices_mask+startIndicesExpanded
print(expanded_indices_mask,expanded_indices_mask.shape)
# raise Exception
expanded_indices_mask=(expanded_indices_mask>=startIndicesExpanded).long()*(expanded_indices_mask<=endIndicesExpanded).long()*expanded_indices_mask.long()



print(passage.shape)
expanded_indices_mask=expanded_indices_mask.unsqueeze(-1).expand(*expanded_indices_mask.shape,passage.size(-1))
print(expanded_indices_mask.shape)

passage=passage.unsqueeze(0).expand(expanded_indices_mask.size(0),*passage.shape)
features=torch.gather(passage,dim=1,index=expanded_indices_mask)
features=features*(features)
print(features.narrow(dim=-1,start =0,length=1))
print(features.shape)
#out B *T^2 *2
