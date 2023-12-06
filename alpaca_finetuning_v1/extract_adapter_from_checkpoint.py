import torch

model = torch.load("./checkpoint/checkpoint-4.pth", map_location="cpu")
new_model = dict()
weight_list = [f"layers.{str(i)}.attention.gate" for i in range(32)]
old_weight_list = [f"layers.{str(i)}.attention.gate" for i in range(32)]
weight_list += ["adapter_query.weight"]

print(weight_list)
print(model["model"]["adapter_query.weight"].shape)

for weight in weight_list:
    new_model[weight] = model["model"][weight]

torch.save(new_model, "adapter_adapter_len10_layer30_epoch5.pth")
