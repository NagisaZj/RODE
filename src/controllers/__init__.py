REGISTRY = {}

from .basic_controller import BasicMAC
from .rode_controller import RODEMAC
from .rode_controller_rnn import RODERNNMAC
from .rode_full_ac_controller import RODEFullACMAC
from .rode_full_ac_controller_rnn import RODEFullACRNNMAC
from .rode_full_ac_noar_controller import RODEFullACNoARMAC
from .rode_controller_comb import RODECombMAC
from .rode_noar_controller import RODENoARMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY['rode_mac'] = RODEMAC
REGISTRY['rode_rnn_mac'] = RODERNNMAC
REGISTRY['rode_full_ac_mac'] = RODEFullACMAC
REGISTRY['rode_full_ac_mac_rnn'] = RODEFullACRNNMAC
REGISTRY['rode_full_ac_noar_mac'] = RODEFullACRNNMAC
REGISTRY['rode_comb_mac'] = RODECombMAC
REGISTRY['rode_noar_mac'] = RODENoARMAC
