"""Predict both cooperative-combo actions from synthetic tensor inputs."""

from __future__ import annotations

import argparse

from reyn import Combo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Directory containing the selected combo release")
    parser.add_argument("--device", default="cpu", help="PyTorch device")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed shared by the two rows")
    args = parser.parse_args()

    model = Combo.load(args.model_dir, device=args.device, seed=args.seed)
    frames, features = model.example_input()
    prediction = model.predict(frames, features)
    print(prediction.action)
    model.reset(seed=args.seed)


if __name__ == "__main__":
    main()
