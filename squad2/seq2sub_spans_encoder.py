import logging
from typing import Any, Dict, List, Optional

import torch
from torch.nn.functional import nll_loss

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1
from allennlp.modules.encoder_base import _EncoderBase
from allennlp.common.registrable import Registrable
from _ast import Tuple
from allennlp.nn.activations import Activation
from allennlp.nn.util import get_lengths_from_binary_sequence_mask,\
    get_mask_from_sequence_lengths
from squad2.utils import getAllSubSpans, BetterTimeDistributed
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from overrides.overrides import overrides

class Seq2SubSpanEncoder(_EncoderBase, Registrable):
    """
    A ``Seq2SeqEncoder`` is a ``Module`` that takes as input a sequence of vectors and returns a
    modified sequence of vectors.  Input shape: ``(batch_size, sequence_length, input_dim)``; output
    shape: ``(batch_size, sequence_length, output_dim)``.

    We add two methods to the basic ``Module`` API: :func:`get_input_dim()` and :func:`get_output_dim()`.
    You might need this if you want to construct a ``Linear`` layer using the output of this encoder,
    or to raise sensible errors for mis-matching input dimensions.
    """
    def get_input_dim(self) -> int:
        """
        Returns the dimension of the vector input for each element in the sequence input
        to a ``Seq2SeqEncoder``. This is `not` the shape of the input tensor, but the
        last element of that shape.
        """
        raise NotImplementedError

    def get_output_dim(self) -> int:
        """
        Returns the dimension of each vector in the sequence output by this ``Seq2SeqEncoder``.
        This is `not` the shape of the returned tensor, but the last element of that shape.
        """
        raise NotImplementedError

    def get_best_answer_span(self,*args) -> int:
        
        raise NotImplementedError
    
class Seq2SpanStartEndIndendentSubSpanEncoder():
    

        
@Seq2SubSpanEncoder.register("seq2subspanencoderdep")
class Seq2SimaltaneousSubSpanEncoderDep(Seq2SubSpanEncoder):
    """
    A ``Seq2SimaltaneousSubSpanEncoder`` encodes logits  seperately for all possible subspans as vecs using provided seq2vec encoder.  As a
    :class:`Seq2VecEncoder`, the input to this module is of shape ``(batch_size, num_tokens,
    input_dim)``, and the output is of shape ``(batch_size, num_sequences,output_dim)``.

    The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
    out a vector of size num_filters. The number of times a convolution layer will be used
    is ``num_tokens - ngram_size + 1``. The corresponding maxpooling layer aggregates all these
    outputs from the convolution layer and outputs the max.

    This operation is repeated for every ngram size passed, and consequently the dimensionality of
    the output after maxpooling is ``len(ngram_filter_sizes) * num_filters``.  This then gets
    (optionally) projected down to a lower dimensional output, specified by ``output_dim``.

    We then use a fully connected layer to project in back to the desired output_dim.  For more
    details, refer to "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural
    Networks for Sentence Classification", Zhang and Wallace 2016, particularly Figure 1.

    Parameters
    ----------
    embedding_dim : ``int``
        This is the input dimension to the encoder.  We need this because we can't do shape
        inference in pytorch, and we need to know what size filters to construct in the CNN.
    num_filters: ``int``
        This is the output dim for each convolutional layer, which is the number of "filters"
        learned by that layer.
    ngram_filter_sizes: ``Tuple[int]``, optional (default=``(2, 3, 4, 5)``)
        This specifies both the number of convolutional layers we will create and their sizes.  The
        default of ``(2, 3, 4, 5)`` will have four convolutional layers, corresponding to encoding
        ngrams of size 2 to 5 with some number of filters.
    conv_layer_activation: ``Activation``, optional (default=``torch.nn.ReLU``)
        Activation to use after the convolution layers.
    output_dim : ``Optional[int]``, optional (default=``None``)
        After doing convolutions and pooling, we'll project the collected features into a vector of
        this size.  If this value is ``None``, we will just return the result of the max pooling,
        giving an output of shape ``len(ngram_filter_sizes) * num_filters``.
    """
    def __init__(self,
                 span_encoder: Seq2VecEncoder,
                 output_dim: Optional[int] = None) -> None:
        super(Seq2SubSpanEncoder, self).__init__()
        self._output_dim=output_dim
        self._span_predictor=TimeDistributed(torch.nn.Linear(output_dim, 1))
        self._span_encoder=span_encoder
        

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, combined_repr: torch.Tensor, passage_lstm_mask: torch.Tensor):  # pylint: disable=arguments-differ
        passage_lengths=get_lengths_from_binary_sequence_mask(passage_lstm_mask)

        #each_answer_feature=B*MaxSubSpans*MaxSpanLength*embedding_dims
        each_answer_features,answer_lengths=getAllSubSpans(combined_repr,passage_lengths,10,padToken=torch.zeros([1]))
        
        '''
            Use the combined representation to simultaneously predict both span start and end.
            We do this by gathering all pairs of indices in the sequence and predicting if that span is the answer 
        '''
        
        #pass each answer feature through gru
        def get_mask_from_sequence_lengths_retriever(maxLength):
            return lambda lengths:get_mask_from_sequence_lengths(lengths.squeeze(-1),maxLength)
        maxLength=torch.max(answer_lengths)
        answer_sequence_mask_creator=TimeDistributed(get_mask_from_sequence_lengths_retriever(maxLength))
        answer_features_mask=answer_sequence_mask_creator(answer_lengths.unsqueeze(-1))
        
        answers_encoded=self._dropout(self._span_encoder(each_answer_features,answer_features_mask))
        
        answer_logits= self._span_predictor(answers_encoded).squeeze(-1)
        answer_sequence_mask_creator=BetterTimeDistributed(util.masked_softmax)

        answer_probs=util.masked_softmax(answer_logits,answer_features_mask.narrow(-1,0,1).squeeze(-1))
        return answer_logits,answer_features_mask,answer_probs,self.get_best_answer_span(answer_probs)
    
    
    def get_best_answer_span(self,answer_probs,batch_to_answer_map) -> int:
        
Seq2SubSpanEncoder.register("seq2subspanencoder")
class Seq2SimaltaneousSubSpanEncoder(Seq2SubSpanEncoder):
    """
    A ``Seq2SimaltaneousSubSpanEncoder`` encodes logits  seperately for all possible subspans as vecs using provided seq2vec encoder.  As a
    :class:`Seq2VecEncoder`, the input to this module is of shape ``(batch_size, num_tokens,
    input_dim)``, and the output is of shape ``(batch_size, num_sequences,output_dim)``.

    The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
    out a vector of size num_filters. The number of times a convolution layer will be used
    is ``num_tokens - ngram_size + 1``. The corresponding maxpooling layer aggregates all these
    outputs from the convolution layer and outputs the max.

    This operation is repeated for every ngram size passed, and consequently the dimensionality of
    the output after maxpooling is ``len(ngram_filter_sizes) * num_filters``.  This then gets
    (optionally) projected down to a lower dimensional output, specified by ``output_dim``.

    We then use a fully connected layer to project in back to the desired output_dim.  For more
    details, refer to "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural
    Networks for Sentence Classification", Zhang and Wallace 2016, particularly Figure 1.

    Parameters
    ----------
    embedding_dim : ``int``
        This is the input dimension to the encoder.  We need this because we can't do shape
        inference in pytorch, and we need to know what size filters to construct in the CNN.
    num_filters: ``int``
        This is the output dim for each convolutional layer, which is the number of "filters"
        learned by that layer.
    ngram_filter_sizes: ``Tuple[int]``, optional (default=``(2, 3, 4, 5)``)
        This specifies both the number of convolutional layers we will create and their sizes.  The
        default of ``(2, 3, 4, 5)`` will have four convolutional layers, corresponding to encoding
        ngrams of size 2 to 5 with some number of filters.
    conv_layer_activation: ``Activation``, optional (default=``torch.nn.ReLU``)
        Activation to use after the convolution layers.
    output_dim : ``Optional[int]``, optional (default=``None``)
        After doing convolutions and pooling, we'll project the collected features into a vector of
        this size.  If this value is ``None``, we will just return the result of the max pooling,
        giving an output of shape ``len(ngram_filter_sizes) * num_filters``.
    """
    def __init__(self,
                 span_encoder: Seq2VecEncoder,
                 output_dim: Optional[int] = None) -> None:
        super(Seq2SubSpanEncoder, self).__init__()
        self._output_dim=output_dim
        self._span_predictor=TimeDistributed(torch.nn.Linear(output_dim, 1))
        self._span_encoder=span_encoder
        

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, combined_repr: torch.Tensor, passage_lstm_mask: torch.Tensor):  # pylint: disable=arguments-differ
        passage_lengths=get_lengths_from_binary_sequence_mask(passage_lstm_mask)

        #each_answer_feature=B*MaxSubSpans*MaxSpanLength*embedding_dims
        each_answer_features,answer_lengths=getAllSubSpans(combined_repr,passage_lengths,10,padToken=torch.zeros([1]))
        
        '''
            Use the combined representation to simultaneously predict both span start and end.
            We do this by gathering all pairs of indices in the sequence and predicting if that span is the answer 
        '''
        
        #pass each answer feature through gru
        def get_mask_from_sequence_lengths_retriever(maxLength):
            return lambda lengths:get_mask_from_sequence_lengths(lengths.squeeze(-1),maxLength)
        maxLength=torch.max(answer_lengths)
        answer_sequence_mask_creator=TimeDistributed(get_mask_from_sequence_lengths_retriever(maxLength))
        answer_features_mask=answer_sequence_mask_creator(answer_lengths.unsqueeze(-1))
        
        answers_encoded=self._dropout(self._span_encoder(each_answer_features,answer_features_mask))
        
        answer_logits= self._span_predictor(answers_encoded).squeeze(-1)
        answer_sequence_mask_creator=BetterTimeDistributed(util.masked_softmax)

        answer_probs=util.masked_softmax(answer_logits,answer_features_mask.narrow(-1,0,1).squeeze(-1))
        return answer_logits,answer_features_mask,answer_probs,self.get_best_answer_span(answer_probs)
    
    def get_best_answer_span(self,answer_probs,batch_to_answer_map) -> in
    