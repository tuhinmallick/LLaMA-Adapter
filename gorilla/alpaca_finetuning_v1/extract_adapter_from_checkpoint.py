import torch
import argparse

args = argparse.ArgumentParser("extract", add_help=False)

args.add_argument("--model_path", type=str)

args = args.parse_args()

model = torch.load(args.model_path, map_location="cpu")
new_model = dict()
weight_list = [f"layers.{str(i)}.attention.gate" for i in range(32)]
old_weight_list = [f"layers.{str(i)}.attention.gate" for i in range(32)]
weight_list += ["adapter_query.weight"]

print(weight_list)
print(model["model"]["adapter_query.weight"].shape)

for weight in weight_list:
    new_model[weight] = model["model"][weight]

save_path = args.model_path.replace('.pth', '-adapter.pth')
torch.save(new_model, save_path)
