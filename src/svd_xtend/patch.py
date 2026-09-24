"""
Patches the SVD_Xtend training script (https://github.com/pixeli99/SVD_Xtend) for our setup.

Two fixes are needed to train on an A100 in bf16:

1. `autocast` is only enabled for fp16 upstream, which disables mixed precision with bf16.
2. Checkpoints are saved from the wrapped model, which writes an almost empty UNet
   (a few hundred bytes instead of ~3 GB). We save the unwrapped model instead.

Usage:
    uv run src/svd_xtend/patch.py --repo ./SVD_Xtend
"""
import argparse
from pathlib import Path

AUTOCAST_OLD = """                        with torch.autocast(
                            str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
                        ):"""

AUTOCAST_NEW = """                        with torch.autocast(
                            str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision in ["fp16", "bf16"]
                        ):"""

SAVE_OLD = """            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()"""

SAVE_NEW = """            for i, model in enumerate(models):
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(
                    os.path.join(output_dir, "unet"),
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                )

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()"""

PATCHES = [("bf16 autocast", AUTOCAST_OLD, AUTOCAST_NEW),
           ("checkpoint saving", SAVE_OLD, SAVE_NEW)]


def apply_patches(repo_dir: str = "./SVD_Xtend") -> str:
    """
    Applies both patches to <repo_dir>/train_svd.py, in place and idempotently.
    Raises RuntimeError if a snippet is neither found nor already patched, which means
    the upstream file changed and the patch must be updated.
    """
    script = Path(repo_dir) / "train_svd.py"
    code = script.read_text()

    for name, old, new in PATCHES:
        if new in code:
            print(f"[{name}] already patched")
        elif old in code:
            code = code.replace(old, new)
            print(f"[{name}] patched")
        else:
            raise RuntimeError(
                f"[{name}] snippet not found in {script}: SVD_Xtend changed upstream, update the patch"
            )

    script.write_text(code)
    return str(script)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch the SVD_Xtend training script")
    parser.add_argument("--repo", type=str, default="./SVD_Xtend", help="Path to the cloned SVD_Xtend repository")
    args = parser.parse_args()
    apply_patches(args.repo)
