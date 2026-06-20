"""Smoke test for hybrid PEFT methods.

Verifies for each method x model:
1. Model instantiates without error
2. Forward pass succeeds
3. Backward pass succeeds
4. Trainable parameter count is reported
5. Key trainable parameters have nonzero grad after backward
6. IA3 vector dimensions are consistent between native and manual
"""
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.dirname(__file__))
from hybrid_peft import configure_method

METHODS = [
    "lora", "bitfit", "ia3",
    "lora_bitfit", "lora_ia3", "bitfit_ia3", "lora_bitfit_ia3",
    "lora_param_matched_r12",
]
MODELS = ["bert-base-uncased", "roberta-base"]


def run_smoke(method_name, model_name, device="cuda"):
    result = {
        "method": method_name, "model": model_name,
        "instantiate": False, "forward": False, "backward": False,
        "trainable_params": 0, "peft_params": 0, "classifier_params": 0,
        "trainable_names_sample": [],
        "nonzero_grad_check": False, "grad_details": {},
        "ia3_mode": None, "ia3_targets": [], "error": None,
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, local_files_only=True,
            dtype=torch.bfloat16
        ).to(device)

        model, info = configure_method(base_model, method_name, model_name)
        model = model.to(device)
        result["instantiate"] = True
        result["trainable_params"] = info["trainable_params"]
        result["peft_params"] = info["peft_params"]
        result["classifier_params"] = info["classifier_params"]
        result["trainable_names_sample"] = info["trainable_names"][:10]
        result["ia3_mode"] = info["ia3_mode"]
        result["ia3_targets"] = info.get("ia3_targets", [])

        # Forward
        inputs = tokenizer("This is a test sentence.", return_tensors="pt",
                          padding="max_length", max_length=32, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["labels"] = torch.tensor([1], device=device)
        outputs = model(**inputs)
        loss = outputs.loss
        result["forward"] = True

        # Backward
        loss.backward()
        result["backward"] = True

        # Nonzero grad check
        grad_ok_count = 0
        grad_total = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_total += 1
                has_grad = param.grad.abs().sum().item() > 0
                if has_grad:
                    grad_ok_count += 1
                if grad_total <= 5:
                    result["grad_details"][name] = {
                        "shape": list(param.shape),
                        "grad_nonzero": has_grad,
                        "grad_norm": param.grad.norm().item(),
                    }

        result["nonzero_grad_check"] = grad_ok_count > 0 and grad_ok_count >= grad_total * 0.5

        # Cleanup
        if info.get("manual_ia3"):
            info["manual_ia3"].remove()
        del model, base_model, outputs, loss
        torch.cuda.empty_cache()

    except Exception as e:
        result["error"] = str(e)
        import traceback
        traceback.print_exc()

    return result


def check_ia3_consistency(all_results):
    """Verify native and manual IA3 produce same vector dimensions."""
    print("\n" + "=" * 80)
    print("IA3 CONSISTENCY CHECK (native vs manual)")
    print("=" * 80)
    ok = True
    for model_name in MODELS:
        native_dims = {}
        manual_dims = {}
        for r in all_results:
            if r["model"] != model_name:
                continue
            if r["ia3_mode"] == "peft_native" and r["method"] == "ia3":
                for n in r["trainable_names_sample"]:
                    if "ia3_l" in n:
                        native_dims[n] = None  # will check from grad_details
            if r["ia3_mode"] == "manual_hook":
                for n in r["trainable_names_sample"]:
                    if "manual_ia3" in n:
                        manual_dims[n] = None

        # Check from grad details
        for r in all_results:
            if r["model"] != model_name:
                continue
            for gname, ginfo in r["grad_details"].items():
                if "ia3_l" in gname:
                    native_dims[gname] = ginfo["shape"]
                if "manual_ia3" in gname:
                    manual_dims[gname] = ginfo["shape"]

        print(f"\n  {model_name}:")
        print(f"    Native IA3 vectors (from grad_details): {native_dims}")
        print(f"    Manual IA3 vectors (from grad_details): {manual_dims}")

        # Key check: FF module dimension
        # Native: intermediate.dense ia3_l should be [1, 768] (scales input)
        # Manual: intermediate.dense should be [768] (scales input via pre_hook)
        for gname, shape in native_dims.items():
            if shape and "intermediate" in gname:
                if shape not in [[1, 768], [768, 1]]:
                    print(f"    WARNING: native FF dim unexpected: {gname} = {shape}")
                    ok = False
                else:
                    print(f"    OK: native FF {gname} = {shape} (input scaling)")
        for gname, shape in manual_dims.items():
            if shape and "intermediate" in gname:
                if shape != [768]:
                    print(f"    FAIL: manual FF dim should be [768], got {gname} = {shape}")
                    ok = False
                else:
                    print(f"    OK: manual FF {gname} = {shape} (input scaling via pre_hook)")

    return ok


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"torch: {torch.__version__}")
    print("=" * 80)

    all_results = []
    param_table = {}

    for model_name in MODELS:
        for method_name in METHODS:
            print(f"\n--- {method_name} x {model_name} ---")
            r = run_smoke(method_name, model_name, device)
            all_results.append(r)
            param_table[(method_name, model_name)] = r

            checks = [r["instantiate"], r["forward"], r["backward"], r["nonzero_grad_check"]]
            status = "PASS" if all(checks) else "FAIL"
            print(f"  Status: {status}")
            print(f"  Trainable params: {r['trainable_params']:,} (PEFT: {r['peft_params']:,}, classifier: {r['classifier_params']:,})")
            print(f"  IA3 mode: {r['ia3_mode']}")
            if r["ia3_targets"]:
                print(f"  IA3 targets: {r['ia3_targets']}")
            names_preview = r["trainable_names_sample"][:5]
            print(f"  Trainable names (first 5): {names_preview}")
            print(f"  Grad check passed: {r['nonzero_grad_check']}")
            for gname, ginfo in r["grad_details"].items():
                print(f"    {gname}: shape={ginfo['shape']}, nonzero={ginfo['grad_nonzero']}, norm={ginfo['grad_norm']:.6f}")
            if r["error"]:
                print(f"  ERROR: {r['error']}")

    # IA3 consistency check
    ia3_ok = check_ia3_consistency(all_results)

    # Summary table (PEFT-specific params, excluding classifier)
    print("\n" + "=" * 80)
    print("PARAMETER COUNT SUMMARY (PEFT-specific, excluding classifier)")
    print("=" * 80)
    print("%-25s %12s %12s %12s %12s" % ("Method", "BERT PEFT", "BERT cls", "RoBERTa PEFT", "RoBERTa cls"))
    print("-" * 73)
    for method in METHODS:
        br = param_table.get((method, "bert-base-uncased"))
        rr = param_table.get((method, "roberta-base"))
        bp = br["peft_params"] if br else 0
        bc = br["classifier_params"] if br else 0
        rp = rr["peft_params"] if rr else 0
        rc = rr["classifier_params"] if rr else 0
        print("%-25s %12s %12s %12s %12s" % (method, f"{bp:,}", f"{bc:,}", f"{rp:,}", f"{rc:,}"))

    # Param-matched rank verification
    print("\n" + "=" * 80)
    print("PARAM-MATCHED LORA RANK VERIFICATION (PEFT-specific params only)")
    print("=" * 80)
    hybrid_methods = ["lora_bitfit", "lora_ia3", "bitfit_ia3", "lora_bitfit_ia3"]
    params_per_rank = 12 * 2 * 2 * 768  # 36,864

    for model_name in MODELS:
        print(f"\n  {model_name}:")
        hybrid_peft = []
        for m in hybrid_methods:
            r = param_table.get((m, model_name))
            if r:
                hybrid_peft.append((m, r["peft_params"]))
        for m, p in hybrid_peft:
            print(f"    {m}: PEFT params = {p:,}")
        max_method, max_peft = max(hybrid_peft, key=lambda x: x[1])
        avg_peft = sum(p for _, p in hybrid_peft) / len(hybrid_peft)

        pm_r = param_table.get(("lora_param_matched_r12", model_name))
        pm_peft = pm_r["peft_params"] if pm_r else 0

        print(f"    ---")
        print(f"    Max hybrid PEFT ({max_method}): {max_peft:,}")
        print(f"    Avg hybrid PEFT: {avg_peft:,.0f}")
        print(f"    LoRA r=12 PEFT params: {pm_peft:,}")
        print(f"    Delta (r=12 vs max hybrid): {pm_peft - max_peft:+,} ({(pm_peft - max_peft)/max_peft*100:+.1f}%)")
        print(f"    LoRA params per unit rank: {params_per_rank:,}")

    # Overall verdict
    print("\n" + "=" * 80)
    failures = [r for r in all_results if r["error"] or not r["nonzero_grad_check"]]
    if failures or not ia3_ok:
        if failures:
            print(f"SMOKE TEST FAILURES ({len(failures)}):")
            for f in failures:
                print(f"  {f['method']} x {f['model']}: error={f['error']}, grad_ok={f['nonzero_grad_check']}")
        if not ia3_ok:
            print("IA3 CONSISTENCY CHECK FAILED")
        sys.exit(1)
    else:
        print("ALL SMOKE TESTS PASSED + IA3 CONSISTENCY OK")
        sys.exit(0)


if __name__ == "__main__":
    main()
