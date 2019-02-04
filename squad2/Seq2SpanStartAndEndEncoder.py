import logging


from allennlp.common.registrable import Registrable
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class Seq2SpanStartAndEndEncoder(torch.nn.module, Registrable):


class Seq2SimaltaneousSpanStartAndEndEncoder(torch.nn.module, Registrable):