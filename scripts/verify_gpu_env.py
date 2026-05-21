import platform

import torch


def main() -> None:
    print(f"python={platform.python_version()}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print(f"gpu={props.name}")
    print(f"compute_capability=sm_{props.major}{props.minor}")
    print(f"total_vram_gb={props.total_memory / (1024 ** 3):.2f}")
    print(f"torch_cuda={torch.version.cuda}")

    if "5090" not in props.name:
        print("warning: GPU name does not look like RTX 5090")
    if (props.major, props.minor) != (12, 0):
        print("warning: expected Blackwell RTX 5090 compute capability sm_120")


if __name__ == "__main__":
    main()

