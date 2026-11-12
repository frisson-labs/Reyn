"""Tensor-only shared and per-Fox language conditioning."""

import argparse

from reyn_doubles import Doubles

parser = argparse.ArgumentParser()
parser.add_argument("model_dir")
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

model = Doubles.load(args.model_dir, device=args.device, team=0)
world, views = model.example_input()
model.say("Attempt a grab.")
prediction = model.predict(world, views)
print(prediction.action.shape)  # [2, 13], one controller per Fox teammate

# Instructions persist until replaced or cleared for that recipient.
model.say("Begin the Fire Fox move.", recipient="fox1")
model.clear(recipient="fox2")
world.frame[...] += 1
prediction = model.predict(world, views)
print(prediction.commands)
model.clear()
model.reset()  # Clears language conditioning and every enemy-view history.
