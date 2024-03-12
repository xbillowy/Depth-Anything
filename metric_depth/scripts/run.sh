# Finetune metric depth-anythig model on multiple datasets
python3 train_mono.py -m zoedepth -d generalizable --pretrained_resource=""

# Test the finetuned metric depth-anything model on datasets with ground truth
python3 visualize.py -m zoedepth -d easyvolcap_test --pretrained_resource local::./depth_anything_finetune/ZoeDepthv1_09-Mar_23-17-8dfdda2d9218_latest.pt

# Visualize on datasets without ground truth
python3 visualize.py -m zoedepth -d easyvolcap_visualize -v --pretrained_resource local::./depth_anything_finetune/ZoeDepthv1_09-Mar_23-17-8dfdda2d9218_latest.pt
