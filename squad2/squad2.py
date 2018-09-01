import json
import logging
from typing import Any,Dict, List, Tuple

from overrides import overrides

from allennlp.common import Params
from collections import Counter
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.fields import Field, TextField, IndexField, MetadataField,LabelField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("squad2")
class Squad2Reader(DatasetReader):
    """
    Reads a JSON-formatted SQuAD file and returns a ``Dataset`` where the ``Instances`` have four
    fields: ``question``, a ``TextField``, ``passage``, another ``TextField``, and ``span_start``
    and ``span_end``, both ``IndexFields`` into the ``passage`` ``TextField``.  We also add a
    ``MetadataField`` that stores the instance's ID, the original passage text, gold answer strings,
    and token offsets into the original passage, accessible as ``metadata['id']``,
    ``metadata['original_passage']``, ``metadata['answer_texts']`` and
    ``metadata['token_offsets']``.  This is so that we can more easily use the official SQuAD
    evaluation script to get metrics.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 maxRows:int = -1) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.maxRows=maxRows
    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        logger.info("Reading the dataset")
        returnedRows=0
        for article in dataset:
            if self.maxRows!=-1 and returnedRows>self.maxRows:
                break
            for paragraph_json in article['paragraphs']:
                paragraph = paragraph_json["context"]
                tokenized_paragraph = self._tokenizer.tokenize(paragraph)                
                for question_answer in paragraph_json['qas']:
                    
                    question_text = question_answer["question"].strip().replace("\n", "")
                    answer_impossible=question_answer["is_impossible"]
                    
                    
                    if answer_impossible:
                        jebaited_texts= [answer['text'] for answer in question_answer['plausible_answers']]
                        jebaited_span_starts = [answer['answer_start'] for answer in question_answer['plausible_answers']]
                        jebaited_span_ends = [start + len(answer) for start, answer in zip(jebaited_span_starts, jebaited_texts)]
                        answer_texts= [""]
                        span_starts = [0]
                        span_ends = [0]
                    else:
                        answer_texts = [answer['text'] for answer in question_answer['answers']]
                        jebaited_texts=[]
                        jebaited_span_starts=[]
                        jebaited_span_ends=[]
                        span_starts = [answer['answer_start'] for answer in question_answer['answers']]
                        span_ends = [start + len(answer) for start, answer in zip(span_starts, answer_texts)]
                    
                    instance = self.text_to_instance(question_text,
                                                     paragraph,
                                                     answer_impossible,
                                                     zip(span_starts, span_ends),
                                                     answer_texts,
                                                     zip(jebaited_span_starts,jebaited_span_ends),
                                                     jebaited_texts,
                                                     tokenized_paragraph)
                    returnedRows+=1
                    yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         answer_impossible:bool,
                         char_spans: List[Tuple[int, int]] = None,
                         answer_texts: List[str] = None,
                         jebaited_char_spans: List[Tuple[int, int]] = None,
                         jebaited_texts: List[str]=None,
                         passage_tokens: List[Token] = None) -> Instance:
        # pylint: disable=arguments-differ
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)
        char_spans = char_spans or []

        # We need to convert character indices in `passage_text` to token indices in
        # `passage_tokens`, as the latter is what we'll actually use for supervision.
        token_spans: List[Tuple[int, int]] = []
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        for char_span_start, char_span_end in char_spans:
            if char_span_start==0 and char_span_end==0:
                (span_start, span_end), error = (0,0),None
            else:
                (span_start, span_end), error = util.char_span_to_token_span(passage_offsets,
                                                                             (char_span_start, char_span_end))                
            if error:
                logger.debug("Passage: %s", passage_text)
                logger.debug("Passage tokens: %s", passage_tokens)
                logger.debug("Question text: %s", question_text)
                logger.debug("Is answer impossible: %s", answer_impossible)
                logger.debug("Answer span: (%d, %d)", char_span_start, char_span_end)
                logger.debug("Token span: (%d, %d)", span_start, span_end)
                logger.debug("Tokens in answer: %s", passage_tokens[span_start:span_end + 1])
                logger.debug("Answer: %s", passage_text[char_span_start:char_span_end])
            token_spans.append((span_start, span_end))

        return make_reading_comprehension_instance(self._tokenizer.tokenize(question_text),
                                                        passage_tokens,
                                                        self._token_indexers,
                                                        passage_text,
                                                        token_spans,
                                                        answer_texts,
                                                        jebaited_texts,
                                                        answer_impossible)
    @classmethod
    def from_params(cls, params: Params) -> 'Squad2Reader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        lazy = params.pop('lazy', False)
        maxRows=params.pop('maxRows',-1)
        params.assert_empty(cls.__name__)
        return cls(tokenizer=tokenizer, token_indexers=token_indexers, lazy=lazy,maxRows=maxRows)

        
def make_reading_comprehension_instance(question_tokens: List[Token],
                                        passage_tokens: List[Token],
                                        token_indexers: Dict[str, TokenIndexer],
                                        passage_text: str,
                                        token_spans: List[Tuple[int, int]] = None,
                                        answer_texts: List[str] = None,
                                        jebait_answer_texts:List[str] = None,
                                        is_answer_jebait:bool=None,
                                        additional_metadata: Dict[str, Any] = None) -> Instance:
    """
    Converts a question, a passage, and an optional answer (or answers) to an ``Instance`` for use
    in a reading comprehension model.

    Creates an ``Instance`` with at least these fields: ``question`` and ``passage``, both
    ``TextFields``; and ``metadata``, a ``MetadataField``.  Additionally, if both ``answer_texts``
    and ``char_span_starts`` are given, the ``Instance`` has ``span_start`` and ``span_end``
    fields, which are both ``IndexFields``.

    Parameters
    ----------
    question_tokens : ``List[Token]``
        An already-tokenized question.
    passage_tokens : ``List[Token]``
        An already-tokenized passage that contains the answer to the given question.
    token_indexers : ``Dict[str, TokenIndexer]``
        Determines how the question and passage ``TextFields`` will be converted into tensors that
        get input to a model.  See :class:`TokenIndexer`.
    passage_text : ``str``
        The original passage text.  We need this so that we can recover the actual span from the
        original passage that the model predicts as the answer to the question.  This is used in
        official evaluation scripts.
    token_spans : ``List[Tuple[int, int]]``, optional
        Indices into ``passage_tokens`` to use as the answer to the question for training.  This is
        a list because there might be several possible correct answer spans in the passage.
        Currently, we just select the most frequent span in this list (i.e., SQuAD has multiple
        annotations on the dev set; this will select the span that the most annotators gave as
        correct).
    answer_texts : ``List[str]``, optional
        All valid answer strings for the given question.  In SQuAD, e.g., the training set has
        exactly one answer per question, but the dev and test sets have several.  TriviaQA has many
        possible answers, which are the aliases for the known correct entity.  This is put into the
        metadata for use with official evaluation scripts, but not used anywhere else.
    additional_metadata : ``Dict[str, Any]``, optional
        The constructed ``metadata`` field will by default contain ``original_passage``,
        ``token_offsets``, ``question_tokens``, ``passage_tokens``, and ``answer_texts`` keys.  If
        you want any other metadata to be associated with each instance, you can pass that in here.
        This dictionary will get added to the ``metadata`` dictionary we already construct.
    """
    additional_metadata = additional_metadata or {}
    fields: Dict[str, Field] = {}
    passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]

    # This is separate so we can reference it later with a known type.
    passage_field = TextField(passage_tokens, token_indexers)
    fields['passage'] = passage_field
    fields['question'] = TextField(question_tokens, token_indexers)
    fields['answer_impossible']= LabelField(1 if is_answer_jebait else 0,"is_jebait_labels",True)
    metadata = {
            'original_passage': passage_text,
            'token_offsets': passage_offsets,
            'question_tokens': [token.text for token in question_tokens],
            'passage_tokens': [token.text for token in passage_tokens],
            }
    if answer_texts:
        metadata['answer_texts'] = answer_texts
    if jebait_answer_texts:
        metadata['jebait_answer_texts']=jebait_answer_texts
    if token_spans is not None:
        if len(token_spans)>0:
            # There may be multiple answer annotations, so we pick the one that occurs the most.  This
            # only matters on the SQuAD dev set, and it means our computed metrics ("start_acc",
            # "end_acc", and "span_acc") aren't quite the same as the official metrics, which look at
            # all of the annotations.  This is why we have a separate official SQuAD metric calculation
            # (the "em" and "f1" metrics use the official script).
            candidate_answers: Counter = Counter()
            for span_start, span_end in token_spans:
                candidate_answers[(span_start, span_end)] += 1
            span_start, span_end = candidate_answers.most_common(1)[0][0]
    
            fields['span_start'] = IndexField(span_start, passage_field)
            fields['span_end'] = IndexField(span_end, passage_field)
        else:
            fields['span_start'] = IndexField(0, passage_field)
            fields['span_end'] = IndexField(0, passage_field)
    else:
        print("Token spans empty")
    metadata.update(additional_metadata)
    fields['metadata'] = MetadataField(metadata)
    return Instance(fields)

print("Squad 2 imported")

if __name__=='__main__':
    reader = Squad2Reader()
    span_lens = []
    for row in reader._read('/home/aman/Documents/squad2/train-v2.0.json'):
        #if row.answer_imp
        if row.fields['answer_impossible']!=1:
            span_len = row.fields['span_end'].sequence_index - row.fields['span_start'].sequence_index +1 
            span_lens.append(span_len)
    print(Counter(span_lens))
