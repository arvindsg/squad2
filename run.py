#!/usr/bin/env python
import logging
import os
import sys
import squad2.squad2
import squad2.bidaf2
import squad2.bidaf
import squad2.squad_filter
if os.environ.get("ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=LEVEL)

from allennlp.commands import main  # pylint: disable=wrong-import-position

if __name__ == "__main__":
    main(prog="allennlp")
