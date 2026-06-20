"""Hybrid PEFT composition logic and manual (IA)3 implementation.

Manual (IA)3 behavior matches PEFT native IA3Config exactly:
- query/value (non-FF): output *= vector[out_features]  (forward hook)
- feedforward (FF): input *= vector[in_features]  (forward PRE-hook)

This ensures identical parameterization whether IA3 is applied via
PEFT native (standalone, bitfit+ia3) or manual hooks (lora+ia3 combos).

Composition semantic for LoRA+IA3: IA3 hooks attach to base_layer (the original
Linear), NOT to the LoRA wrapper. Therefore IA3 scales only the base projection
output; the LoRA additive update (lora_B @ lora_A @ x) is NOT scaled by IA3.
Final output = IA3(base_layer(x)) + lora_B(lora_A(x)).
"""
from __future__ import annotations
import torch
import torch.nn as nn
from peft import LoraConfig, IA3Config, get_peft_model, TaskType


class ManualIA3:
    """Injects (IA)3 scaling vectors via forward hooks matching PEFT native behavior.
    
    Non-FF modules: output *= learned_vector (shape=[out_features])
    FF modules: input *= learned_vector (shape=[in_features])
    
    Skips LoRA internal sub-modules (lora_A, lora_B).
    """

    def __init__(self, model: nn.Module, target_modules: list[str], ff_modules: list[str]):
        self.hooks = []
        self.vectors = nn.ParameterDict()
        self.target_modules = target_modules
        self.ff_modules = ff_modules
        self.injected_modules = []
        self._inject(model)

    def _inject(self, model: nn.Module):
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if "lora_A" in name or "lora_B" in name or "lora_embedding" in name:
                continue
            is_ff = any(t in name for t in self.ff_modules)
            is_target = any(t in name for t in self.target_modules)
            if not (is_ff or is_target):
                continue

            param_key = name.replace(".", "_")

            if is_ff:
                # FF modules: scale input, vector dim = in_features (matches native [1, in_features])
                dim = module.in_features
                vec = nn.Parameter(torch.ones(dim, device=next(module.parameters()).device,
                                             dtype=next(module.parameters()).dtype))
                self.vectors[param_key] = vec

                def make_pre_hook(v):
                    def hook_fn(mod, args):
                        x = args[0] if isinstance(args, tuple) else args
                        return (x * v,) + args[1:] if isinstance(args, tuple) and len(args) > 1 else (x * v,)
                    return hook_fn

                h = module.register_forward_pre_hook(make_pre_hook(vec))
            else:
                # Non-FF (query/value): scale output, vector dim = out_features (matches native [out_features, 1])
                dim = module.out_features
                vec = nn.Parameter(torch.ones(dim, device=next(module.parameters()).device,
                                             dtype=next(module.parameters()).dtype))
                self.vectors[param_key] = vec

                def make_post_hook(v):
                    def hook_fn(mod, inp, out):
                        return out * v
                    return hook_fn

                h = module.register_forward_hook(make_post_hook(vec))

            self.hooks.append(h)
            self.injected_modules.append((name, "pre_hook/input" if is_ff else "post_hook/output", dim))

    def get_trainable_params(self):
        return [(k, v) for k, v in self.vectors.items()]

    def get_injection_report(self):
        return self.injected_modules

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# Target module names (same for BERT and RoBERTa at base size)
# For IA3Config: feedforward_modules must be subset of target_modules
IA3_TARGET_MODULES = ["query", "value", "intermediate.dense"]
IA3_FF_MODULES = ["intermediate.dense"]

LORA_TARGETS = ["query", "value"]


def configure_method(model, method_name: str, model_name: str):
    """Apply PEFT method(s) to model. Returns (model, info_dict)."""
    info = {"ia3_mode": None, "ia3_targets": [], "manual_ia3": None}

    if method_name == "lora":
        config = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0,
            target_modules=LORA_TARGETS,
            task_type=TaskType.SEQ_CLS, bias="none"
        )
        model = get_peft_model(model, config)

    elif method_name == "bitfit":
        for name, param in model.named_parameters():
            param.requires_grad = "bias" in name or "classifier" in name

    elif method_name == "ia3":
        config = IA3Config(
            target_modules=IA3_TARGET_MODULES,
            feedforward_modules=IA3_FF_MODULES,
            task_type=TaskType.SEQ_CLS
        )
        model = get_peft_model(model, config)
        info["ia3_mode"] = "peft_native"

    elif method_name == "lora_bitfit":
        config = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0,
            target_modules=LORA_TARGETS,
            task_type=TaskType.SEQ_CLS, bias="none"
        )
        model = get_peft_model(model, config)
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True

    elif method_name == "lora_ia3":
        config = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0,
            target_modules=LORA_TARGETS,
            task_type=TaskType.SEQ_CLS, bias="none"
        )
        model = get_peft_model(model, config)
        manual_ia3 = ManualIA3(model, ["query", "value"], ["intermediate.dense"])
        for pname, param in manual_ia3.get_trainable_params():
            model.register_parameter(f"manual_ia3_{pname}", param)
            param.requires_grad = True
        info["ia3_mode"] = "manual_hook"
        info["ia3_targets"] = ["query", "value", "intermediate.dense"]
        info["manual_ia3"] = manual_ia3

    elif method_name == "bitfit_ia3":
        config = IA3Config(
            target_modules=IA3_TARGET_MODULES,
            feedforward_modules=IA3_FF_MODULES,
            task_type=TaskType.SEQ_CLS
        )
        model = get_peft_model(model, config)
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
        info["ia3_mode"] = "peft_native"

    elif method_name == "lora_bitfit_ia3":
        config = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0,
            target_modules=LORA_TARGETS,
            task_type=TaskType.SEQ_CLS, bias="none"
        )
        model = get_peft_model(model, config)
        manual_ia3 = ManualIA3(model, ["query", "value"], ["intermediate.dense"])
        for pname, param in manual_ia3.get_trainable_params():
            model.register_parameter(f"manual_ia3_{pname}", param)
            param.requires_grad = True
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
        info["ia3_mode"] = "manual_hook"
        info["ia3_targets"] = ["query", "value", "intermediate.dense"]
        info["manual_ia3"] = manual_ia3

    elif method_name.startswith("lora_param_matched"):
        rank = int(method_name.split("_r")[-1])
        config = LoraConfig(
            r=rank, lora_alpha=rank * 2, lora_dropout=0.0,
            target_modules=LORA_TARGETS,
            task_type=TaskType.SEQ_CLS, bias="none"
        )
        model = get_peft_model(model, config)

    else:
        raise ValueError(f"Unknown method: {method_name}")

    # Count trainable params
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    info["trainable_params"] = sum(p.numel() for _, p in trainable)
    info["trainable_names"] = [n for n, _ in trainable]

    # Decompose: PEFT-specific vs classifier
    classifier_params = sum(p.numel() for n, p in trainable if "classifier" in n)
    info["classifier_params"] = classifier_params
    info["peft_params"] = info["trainable_params"] - classifier_params

    return model, info
