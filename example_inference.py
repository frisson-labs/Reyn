"""Predict an action from synthetic model inputs and an English phrase."""

from __future__ import annotations

import argparse

from reyn import Reyn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Directory containing compatible model weights")
    parser.add_argument("--phrase", default="Go to the left side.", help="English language conditioning")
    parser.add_argument("--device", default="cpu", help="PyTorch device, such as cpu or cuda")
    parser.add_argument("--character", default="fox", help="Character key supported by the selected model")
    parser.add_argument("--stage", default="BF", choices=("FD", "BF", "DL", "YS", "FoD", "PS"))
    args = parser.parse_args()

    from reyn.language.stages import STAGES

    model = Reyn.load(args.model_dir, character=args.character, device=args.device)
    inputs = model.example_input()  # Synthetic tensors illustrate the input structure.
    inputs["stage"].fill_(STAGES[args.stage].model_id)
    prediction = model.predict(inputs, phrase=args.phrase)

    print("action:", prediction.action)


if __name__ == "__main__":
    main()
