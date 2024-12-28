import flax.linen as nn
from .vdn import VDN
from .qmix import QMIX
from .qfix import QFix, AdditiveQFix
from .qfix_sum_alt import QFixSumAlt, AdditiveQFixSumAlt


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


def make_fixer(config, num_agents: int) -> nn.Module:
    config_qfix = config["QFIX"]

    fixer = config_qfix["FIXER"]
    is_additive = fixer.startswith("q+fix")

    w_delta = config_qfix.get("W_DELTA", 0.0)
    w_gt = config_qfix.get("W_GT", -1.0 if is_additive else 0.0)
    detach_advantages = config_qfix.get("DETACH_ADVANTAGES", True)

    if fixer == "qfix-sum-alt":
        return QFixSumAlt(
            hidden_size=config["HIDDEN_SIZE"],
            num_agents=num_agents,
            w_delta=w_delta,
            w_gt=w_gt,
        )

    if fixer == "q+fix-sum-alt":
        return AdditiveQFixSumAlt(
            hidden_size=config["HIDDEN_SIZE"],
            num_agents=num_agents,
            w_delta=w_delta,
            w_gt=w_gt,
            detach_advantages=detach_advantages,
        )

    fixee = make_fixee(config)

    if fixer == "qfix":
        return QFix(
            hidden_size=config["HIDDEN_SIZE"],
            fixee=fixee,
            w_delta=w_delta,
            w_gt=w_gt,
        )

    if fixer == "q+fix":
        return AdditiveQFix(
            hidden_size=config["HIDDEN_SIZE"],
            fixee=fixee,
            w_delta=w_delta,
            w_gt=w_gt,
            detach_advantages=detach_advantages,
        )

    raise ValueError(f"Invalid fixer name {fixer}")
