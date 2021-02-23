REGISTRY = {}

from .dot_selector import DotSelector
from .q_selector import QSelector
from .dot_rnn_selector import DotRNNSelector
REGISTRY['dot'] = DotSelector
REGISTRY['dot_rnn'] = DotRNNSelector
REGISTRY['q'] = QSelector
