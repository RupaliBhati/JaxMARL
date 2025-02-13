from typing import cast

import flax.linen as nn

from .adapters import FFAdapter
from .protocol import QFixProtocol
from .qfix import AdditiveQFix, QFix
from .qfix_lin import AdditiveQFixLin, QFixLin
from .qmix import QMIX
from .vdn import VDN


def make_fixee(config) -> nn.Module:
    fixee = config["QFIX"]["FIXEE"]

    if fixee == "vdn":
        return VDN()

    if fixee == "qmix":
        return QMIX(
            config["MIXER_EMBEDDING_DIM"],
            config["MIXER_HYPERNET_HIDDEN_DIM"],
            config["MIXER_INIT_SCALE"],
        )

    raise ValueError(f"Invalid {fixee=}")


def make_fixer(config, num_agents: int, *, wrap_ff_adapter=False) -> nn.Module:
    if wrap_ff_adapter:
        fixer = cast(QFixProtocol, make_fixer(config, num_agents))
        return FFAdapter(fixer)

    config_qfix = config["QFIX"]

    fixer = config_qfix["FIXER"]
    is_additive = fixer.startswith("q+fix")

    w_delta = config_qfix.get("W_DELTA", 0.0)
    w_gt = config_qfix.get("W_GT", -1.0 if is_additive else 0.0)
    detach_advantages = config_qfix.get("DETACH_ADVANTAGES", True)
    debug_recover_fixee =config_qfix.get("DEBUG_RECOVER_FIXEE", False)

    if fixer == "qfix-lin":
        return QFixLin(
            hidden_size=config["HIDDEN_SIZE"],
            num_agents=num_agents,
            w_delta=w_delta,
            w_gt=w_gt,
            debug_recover_fixee=debug_recover_fixee,
        )

    if fixer == "q+fix-lin":
        return AdditiveQFixLin(
            hidden_size=config["HIDDEN_SIZE"],
            num_agents=num_agents,
            w_delta=w_delta,
            w_gt=w_gt,
            detach_advantages=detach_advantages,
            debug_recover_fixee=debug_recover_fixee,
        )

    fixee = make_fixee(config)

    if fixer == "qfix":
        return QFix(
            hidden_size=config["HIDDEN_SIZE"],
            fixee=fixee,
            w_delta=w_delta,
            w_gt=w_gt,
            debug_recover_fixee=debug_recover_fixee,
        )

    if fixer == "q+fix":
        return AdditiveQFix(
            hidden_size=config["HIDDEN_SIZE"],
            fixee=fixee,
            w_delta=w_delta,
            w_gt=w_gt,
            detach_advantages=detach_advantages,
            debug_recover_fixee=debug_recover_fixee,
        )

    raise ValueError(f"Invalid fixer name {fixer}")
