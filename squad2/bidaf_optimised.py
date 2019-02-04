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
from squad2.utils import getAllSubSpans, BetterTimeDistributed, masked_softmax,\
    getIndiceForGoldSubSpan, getSpanStarts, getSpanEnds,\
    get_best_answers_mask_over_passage
from allennlp.nn.util import get_lengths_from_binary_sequence_mask,\
    get_mask_from_sequence_lengths
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("bidafsimal")
class BidirectionalAttentionFlowSimal(Model):
    """
    This class implements Minjoon Seo's `Bidirectional Attention Flow model
    <https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Seo-Kembhavi/7586b7cca1deba124af80609327395e613a20e9d>`_
    for answering reading comprehension questions (ICLR 2017).

    The basic layout is pretty simple: encode words as a combination of word embeddings and a
    character-level encoder, pass the word representations through a bi-LSTM/GRU, use a matrix of
    attentions to put question information into the passage word representations (this is the only
    part that is at all non-standard), pass this through another few layers of bi-LSTMs/GRUs, and
    do a softmax over span start and span end.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
    num_highway_layers : ``int``
        The number of highway layers to use in between embedding the input and passing it through
        the phrase layer.
    phrase_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and doing the bidirectional attention.
    attention_similarity_function : ``SimilarityFunction``
        The similarity function that we will use when comparing encoded passage and question
        representations.
    modeling_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between the bidirectional
        attention and predicting span start and end.
    span_end_encoder : ``Seq2SeqEncoder``
        The encoder that we will use to incorporate span start predictions into the passage state
        before predicting span end.
    dropout : ``float``, optional (default=0.2)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    mask_lstms : ``bool``, optional (default=True)
        If ``False``, we will skip passing the mask to the LSTM layers.  This gives a ~2x speedup,
        with only a slight performance decrease, if any.  We haven't experimented much with this
        yet, but have confirmed that we still get very similar performance with much faster
        training times.  We still use the mask for all softmaxes, but avoid the shuffling that's
        required when using masking with pytorch LSTMs.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 num_highway_layers: int,
                 phrase_layer: Seq2SeqEncoder,
                 attention_similarity_function: SimilarityFunction,
                 modeling_layer: Seq2SeqEncoder,
                 span_encoder: Seq2VecEncoder,
                 dropout: float = 0.2,
                 mask_lstms: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BidirectionalAttentionFlowSimal, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._highway_layer = TimeDistributed(Highway(text_field_embedder.get_output_dim(),
                                                      num_highway_layers))
        self._phrase_layer = phrase_layer
        self._matrix_attention = LegacyMatrixAttention(attention_similarity_function)
        self._modeling_layer = modeling_layer
        self._span_encoder = TimeDistributed(span_encoder) 
        modeling_dim = modeling_layer.get_output_dim()
        encoding_dim = phrase_layer.get_output_dim()
        output_dim=span_encoder.get_output_dim()
        self._span_predictor=TimeDistributed(torch.nn.Linear(output_dim, 1))

        # Bidaf has lots of layer dimensions which need to match up - these aren't necessarily
        # obvious from the configuration files, so we check here.
        check_dimensions_match(modeling_layer.get_input_dim(), 4 * encoding_dim,
                               "modeling layer input dim", "4 * encoding dim")
        check_dimensions_match(text_field_embedder.get_output_dim(), phrase_layer.get_input_dim(),
                               "text field embedder output dim", "phrase layer input dim")
        check_dimensions_match(span_encoder.get_input_dim(), 4 * encoding_dim +  modeling_dim,
                               "span end encoder input dim", "4 * encoding dim + 3 * modeling dim")
        
        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._squad_metrics = SquadEmAndF1()
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._mask_lstms = mask_lstms

        initializer(self)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``.
        passage : Dict[str, torch.LongTensor]
            From a ``TextField``.  The model assumes that this passage contains the answer to the
            question, and predicts the beginning and ending positions of the answer within the
            passage.
        span_start : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            beginning position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        span_end : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            ending position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the question ID, original passage text, and token
            offsets into the passage for each instance in the batch.  We use this for computing
            official metrics using the official SQuAD evaluation script.  The length of this list
            should be the batch size, and each dictionary should have the keys ``id``,
            ``original_passage``, and ``token_offsets``.  If you only want the best span string and
            don't care about official metrics, you can omit the ``id`` key.

        Returns
        -------
        An output dictionary consisting of:
        span_start_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span start position.
        span_start_probs : torch.FloatTensor
            The result of ``softmax(span_start_logits)``.
        span_end_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span end position (inclusive).
        span_end_probs : torch.FloatTensor
            The result of ``softmax(span_end_logits)``.
        best_span : torch.IntTensor
            The result of a constrained inference over ``span_start_logits`` and
            ``span_end_logits`` to find the most probable span.  Shape is ``(batch_size, 2)``
            and each offset is a token index.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        best_span_str : List[str]
            If sufficient metadata was provided for the instances in the batch, we also return the
            string from the original passage that the model thinks is the best answer to the
            question.
        """
        embedded_question = self._highway_layer(self._text_field_embedder(question))
        embedded_passage = self._highway_layer(self._text_field_embedder(passage))
        batch_size = embedded_question.size(0)
        passage_length = embedded_passage.size(1)
        question_mask = util.get_text_field_mask(question).float()
        passage_mask = util.get_text_field_mask(passage).float()
        question_lstm_mask = question_mask if self._mask_lstms else None
        passage_lstm_mask = passage_mask if self._mask_lstms else None

        encoded_question = self._dropout(self._phrase_layer(embedded_question, question_lstm_mask))
        encoded_passage = self._dropout(self._phrase_layer(embedded_passage, passage_lstm_mask))
        encoding_dim = encoded_question.size(-1)

        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = util.last_dim_softmax(passage_question_similarity, question_mask)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)

        # We replace masked values with something really negative here, so they don't affect the
        # max below.
        masked_similarity = util.replace_masked_values(passage_question_similarity,
                                                       question_mask.unsqueeze(1),
                                                       -1e7)
        # Shape: (batch_size, passage_length)
        question_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        # Shape: (batch_size, passage_length)
        question_passage_attention = util.masked_softmax(question_passage_similarity, passage_mask)
        # Shape: (batch_size, encoding_dim)
        question_passage_vector = util.weighted_sum(encoded_passage, question_passage_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(batch_size,
                                                                                    passage_length,
                                                                                    encoding_dim)

        # Shape: (batch_size, passage_length, encoding_dim * 4)
        final_merged_passage = torch.cat([encoded_passage,
                                          passage_question_vectors,
                                          encoded_passage * passage_question_vectors,
                                          encoded_passage * tiled_question_passage_vector],
                                         dim=-1)

        modeled_passage = self._dropout(self._modeling_layer(final_merged_passage, passage_lstm_mask))
        modeling_dim = modeled_passage.size(-1)
        
        
        combined_repr = torch.cat([final_merged_passage,modeled_passage],dim=-1)
        passage_lengths=get_lengths_from_binary_sequence_mask(passage_lstm_mask)

        each_answer_features,answer_lengths,answer_features_over_passage_mask=getAllSubSpans(combined_repr,passage_lengths,10)

        
        #each_answer_feature=B*MaxSubSpans*MaxSpanLength*embedding_dims
        each_answer_features,answer_lengths,answer_features_over_passage_mask=getAllSubSpans(combined_repr,passage_lengths,10)
        
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
        
        del answer_sequence_mask_creator
        
        answers_encoded=self._dropout(self._span_encoder(each_answer_features,answer_features_mask))
        
        answer_logits= self._span_predictor(answers_encoded).squeeze(-1)
#         answer_sequence_mask_creator=BetterTimeDistributed(masked_softmax)
        valid_answers_mask=answer_features_mask.narrow(-1,0,1).squeeze(-1)
        
        del answer_features_mask
        
        answer_probs=masked_softmax(answer_logits,valid_answers_mask)
        
        best_span_answer_probs_indice=self.get_best_span(answer_probs)
        
        best_answers_mask=get_best_answers_mask_over_passage(answer_features_over_passage_mask,best_span_answer_probs_indice)
        best_span_start=getSpanStarts(best_answers_mask)
        best_span_end=getSpanEnds(best_answers_mask)
        best_span=torch.stack([best_span_start,best_span_end],dim=-1)

        output_dict = {
        "passage_question_attention": passage_question_attention,
#         "span_start_logits": span_start_logits,
#         "span_start_probs": span_start_probs,
#         "span_end_logits": span_end_logits,
#         "span_end_probs": span_end_probs,
        "answer_probs":answer_probs,
        "best_span": best_span
        }
    #         best_span = self.get_best_span(span_start_probs,span_end_probs)

        
#         # Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim))
#         span_start_input = self._dropout(torch.cat([final_merged_passage, modeled_passage], dim=-1))
#         # Shape: (batch_size, passage_length)
#         span_start_logits = self._span_start_predictor(span_start_input).squeeze(-1)
#         # Shape: (batch_size, passage_length)
#         span_start_probs = util.masked_softmax(span_start_logits, passage_mask)
# 
#         # Shape: (batch_size, modeling_dim)
#         span_start_representation = util.weighted_sum(modeled_passage, span_start_probs)
#         # Shape: (batch_size, passage_length, modeling_dim)
#         tiled_start_representation = span_start_representation.unsqueeze(1).expand(batch_size,
#                                                                                    passage_length,
#                                                                                    modeling_dim)
# 
#         # Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim * 3)
#         span_end_representation = torch.cat([final_merged_passage,
#                                              modeled_passage,
#                                              tiled_start_representation,
#                                              modeled_passage * tiled_start_representation],
#                                             dim=-1)
#         # Shape: (batch_size, passage_length, encoding_dim)
#         encoded_span_end = self._dropout(self._span_end_encoder(span_end_representation,
#                                                                 passage_lstm_mask))
#         # Shape: (batch_size, passage_length, encoding_dim * 4 + span_end_encoding_dim)
#         span_end_input = self._dropout(torch.cat([final_merged_passage, encoded_span_end], dim=-1))
#         span_end_logits = self._span_end_predictor(span_end_input).squeeze(-1)
#         span_end_probs = util.masked_softmax(span_end_logits, passage_mask)
#         span_start_logits = util.replace_masked_values(span_start_logits, passage_mask, -1e7)
#         span_end_logits = util.replace_masked_values(span_end_logits, passage_mask, -1e7)
#         best_span = self.get_best_span(span_start_logits, span_end_logits)

#         output_dict = {
#                 "passage_question_attention": passage_question_attention,
#                 "span_start_logits": span_start_logits,
#                 "span_start_probs": span_start_probs,
#                 "span_end_logits": span_end_logits,
#                 "span_end_probs": span_end_probs,
#                 "best_span": best_span,
#                 }

        # Compute the loss for training.
        if span_start is not None:
#             loss = nll_loss(util.masked_log_softmax(span_start_logits, passage_mask), span_start.squeeze(-1))
#             self._span_start_accuracy(span_start_logits, span_start.squeeze(-1))
#             loss += nll_loss(util.masked_log_softmax(span_end_logits, passage_mask), span_end.squeeze(-1))
#             self._span_end_accuracy(span_end_logits, span_end.squeeze(-1))
            self._span_accuracy(best_span, torch.stack([span_start, span_end], -1))
#             print(answer_logits.shape,answer_features_mask.shape)
            
            
            loss=nll_loss(util.masked_log_softmax(answer_logits,valid_answers_mask),getIndiceForGoldSubSpan(span_start,span_end,answer_features_over_passage_mask))
            
            del answer_features_over_passage_mask
            output_dict["loss"] = loss

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        if metadata is not None:
            output_dict['best_span_str'] = []
            question_tokens = []
            passage_tokens = []
            for i in range(batch_size):
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_tokens'])
                passage_str = metadata[i]['original_passage']
                try:
                    print (passage_str.encode('ascii', 'ignore'))
                except :
                    print("Error")
                offsets = metadata[i]['token_offsets']
                predicted_span = tuple(best_span[i].detach().cpu().numpy())
                start_offset = offsets[predicted_span[0]][0]
                end_offset = offsets[predicted_span[1]][1]
                best_span_string = passage_str[start_offset:end_offset]
                output_dict['best_span_str'].append(best_span_string)
                answer_texts = metadata[i].get('answer_texts', [])
                if answer_texts:
                    self._squad_metrics(best_span_string, answer_texts)
            output_dict['question_tokens'] = question_tokens
            output_dict['passage_tokens'] = passage_tokens
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        return {
#                 'start_acc': self._span_start_accuracy.get_metric(reset),
#                 'end_acc': self._span_end_accuracy.get_metric(reset),
                'span_acc': self._span_accuracy.get_metric(reset),
                'em': exact_match,
                'f1': f1_score,
                }

    @staticmethod
    def get_best_span(span_probs: torch.Tensor) -> torch.Tensor:
        if span_probs.dim() != 2:
            raise ValueError("Input shape must be (batch_size, num_answers)")
        return torch.argmax(span_probs, 1)
        
#     @classmethod
#     def from_params(cls, vocab: Vocabulary, params: Params) -> 'BidirectionalAttentionFlow':
#         embedder_params = params.pop("text_field_embedder")
#         text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
#         num_highway_layers = params.pop_int("num_highway_layers")
#         phrase_layer = Seq2SeqEncoder.from_params(params.pop("phrase_layer"))
#         similarity_function = SimilarityFunction.from_params(params.pop("similarity_function"))
#         modeling_layer = Seq2SeqEncoder.from_params(params.pop("modeling_layer"))
#         span_end_encoder = Seq2SeqEncoder.from_params(params.pop("span_end_encoder"))
#         dropout = params.pop_float('dropout', 0.2)
# 
#         initializer = InitializerApplicator.from_params(params.pop('initializer', []))
#         regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
# 
#         mask_lstms = params.pop_bool('mask_lstms', True)
#         params.assert_empty(cls.__name__)
#         return cls(vocab=vocab,
#                    text_field_embedder=text_field_embedder,
#                    num_highway_layers=num_highway_layers,
#                    phrase_layer=phrase_layer,
#                    attention_similarity_function=similarity_function,
#                    modeling_layer=modeling_layer,
#                    span_end_encoder=span_end_encoder,
#                    dropout=dropout,
#                    mask_lstms=mask_lstms,
#                    initializer=initializer,
#                    regularizer=regularizer)
