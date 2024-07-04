import torch

best_model_state_dict = torch.load(join(args.save_dir, "/scratch/ds5725/deep-visual-geo-localization-benchmark/logs/default/2023-04-22_19-42-06/best_model.pth"))["model_state_dict"]

model.load_state_dict(best_model_state_dict)
