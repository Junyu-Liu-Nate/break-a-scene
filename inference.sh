# python train.py \
#   --instance_data_dir examples/creature  \
#   --num_of_assets 3 \
#   --initializer_tokens creature bowl stone \
#   --class_data_dir inputs/data_dir \
#   --phase1_train_steps 400 \
#   --phase2_train_steps 400 \
#   --output_dir outputs/creature_debug \
#   --no_prior_preservation
#   --log_checkpoints
# python inference.py   --model_path "outputs/creature_debug"   --prompt "a photo of <asset0> on the street"   --output_path "outputs/creature_inference/asset0.jpg"
# python inference.py   --model_path "outputs/creature_debug"   --prompt "a photo of <asset1> on the street"   --output_path "outputs/creature_inference/asset1.jpg"
# python inference.py   --model_path "outputs/creature_debug"   --prompt "a photo of <asset2> on the street"   --output_path "outputs/creature_inference/asset2.jpg"
# python inference.py   --model_path "outputs/creature_debug"   --prompt "a photo of <asset0> and <asset1> on the street"   --output_path "outputs/creature_inference/asset0_asset1.jpg"
# python inference.py   --model_path "outputs/creature_debug"   --prompt "a photo of <asset1> and <asset2> on the street"   --output_path "outputs/creature_inference/asset1_asset2.jpg"
# python inference.py   --model_path "outputs/creature_debug"   --prompt "a photo of <asset0> and <asset2> on the street"   --output_path "outputs/creature_inference/asset0_asset2.jpg"
# python inference.py   --model_path "outputs/creature_debug"   --prompt "a photo of <asset0> and <asset1> and <asset2> on the street"   --output_path "outputs/creature_inference/asset0_asset1_asset2.jpg"

# python inference.py   --model_path "outputs/chair_two"   --prompt "a photo of a chair with <asset0> on the street"   --output_path "outputs/chair_two_inference/asset0.jpg"
# python inference.py   --model_path "outputs/chair_two"   --prompt "a photo of a chair with <asset1> on the street"   --output_path "outputs/chair_two_inference/asset1.jpg"
# python inference.py   --model_path "outputs/chair_two"   --prompt "a photo of a chair with <asset2> on the street"   --output_path "outputs/chair_two_inference/asset2.jpg"
# python inference.py   --model_path "outputs/chair_two"   --prompt "a photo of a chair with <asset3> on the street"   --output_path "outputs/chair_two_inference/asset3.jpg"
# python inference.py   --model_path "outputs/chair_two"   --prompt "a photo of a chair with <asset0> and <asset1> on the street"   --output_path "outputs/chair_two_inference/asset0_asset1.jpg"
# python inference.py   --model_path "outputs/chair_two"   --prompt "a photo of a chair with <asset1> and <asset2> on the street"   --output_path "outputs/chair_two_inference/asset1_asset2.jpg"
# python inference.py   --model_path "outputs/chair_two"   --prompt "a photo of a chair with <asset2> and <asset3> on the street"   --output_path "outputs/chair_two_inference/asset2_asset3.jpg"
# python inference.py   --model_path "outputs/chair_two"   --prompt "a photo of a chair with <asset0> and <asset1> and <asset2> and <asset3> on the street"   --output_path "outputs/chair_two_inference/asset0_asset1_asset2_asset3.jpg"

### Single objects
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset0> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset0.jpg"
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset1> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset1.jpg"
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset2> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset2.jpg"
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset3> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset3.jpg"
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset4> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset4.jpg"
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset5> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset5.jpg"
### Two objects
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset0> and <asset1> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset0_asset1.jpg"
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset0> and <asset2> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset0_asset2.jpg"
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset0> and <asset3> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset0_asset3.jpg"
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset0> and <asset4> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset0_asset4.jpg"
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset0> and <asset5> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset0_asset5.jpg"
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset1> and <asset2> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset1_asset2.jpg"
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset1> and <asset3> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset1_asset3.jpg"
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset1> and <asset4> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset1_asset4.jpg"
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset1> and <asset5> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset1_asset5.jpg"
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset2> and <asset3> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset2_asset3.jpg"
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset2> and <asset4> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset2_asset4.jpg"
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset2> and <asset5> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset2_asset5.jpg"
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset3> and <asset4> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset3_asset4.jpg"
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset3> and <asset5> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset3_asset5.jpg"
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset4> and <asset5> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset4_asset5.jpg"
### three objects
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset0> and <asset1> and <asset2> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset0_asset1_asset2.jpg"
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset1> and <asset2> and <asset3> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset1_asset2_asset3.jpg"
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset2> and <asset3> and <asset4> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset2_asset3_asset4.jpg"
python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset3> and <asset4> and <asset5> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset3_asset4_asset5.jpg"
# python inference.py   --model_path "outputs/creature_toys_2024-10-17-16-17"   --prompt "a photo of <asset0> and <asset1> and <asset2> and <asset3> and <asset4> and <asset5> on the street"   --output_path "outputs/creature_toys_2024-10-17-16-17_inference/asset0_asset1_asset2_asset3_asset4_asset5.jpg"

# ### Recons original
# python inference.py   --model_path "outputs/chair_two"   --prompt "a photo of a chair with <asset0> and <asset1> and <asset2> and <asset3> on the street"   --output_path "outputs/chair_two_inference/asset0_asset1_asset2_asset3.jpg"
# python inference.py   --model_path "outputs/chair_two"   --prompt "a photo of a chair with <asset4> and <asset5> and <asset6> and <asset7> on the street"   --output_path "outputs/chair_two_inference/asset4_asset5_asset6_asset7.jpg"
# ### Swap 1 part from img0 by a part from img 1
# python inference.py   --model_path "outputs/chair_two"   --prompt "a photo of a chair with <asset4> and <asset1> and <asset2> and <asset3> on the street"   --output_path "outputs/chair_two_inference/asset4_asset1_asset2_asset3.jpg"
# python inference.py   --model_path "outputs/chair_two"   --prompt "a photo of a chair with <asset0> and <asset5> and <asset2> and <asset3> on the street"   --output_path "outputs/chair_two_inference/asset0_asset5_asset2_asset3.jpg"
# python inference.py   --model_path "outputs/chair_two"   --prompt "a photo of a chair with <asset0> and <asset1> and <asset6> and <asset3> on the street"   --output_path "outputs/chair_two_inference/asset0_asset1_asset6_asset3.jpg"
# python inference.py   --model_path "outputs/chair_two"   --prompt "a photo of a chair with <asset0> and <asset1> and <asset2> and <asset7> on the street"   --output_path "outputs/chair_two_inference/asset0_asset1_asset2_asset7.jpg"
# ### Swap 2 parts from img0 by 2 parts from img 1
# python inference.py   --model_path "outputs/chair_two"   --prompt "a photo of a chair with <asset4> and <asset5> and <asset2> and <asset3> on the street"   --output_path "outputs/chair_two_inference/asset4_asset5_asset2_asset3.jpg"
# python inference.py   --model_path "outputs/chair_two"   --prompt "a photo of a chair with <asset0> and <asset5> and <asset6> and <asset3> on the street"   --output_path "outputs/chair_two_inference/asset0_asset5_asset7_asset3.jpg"
# python inference.py   --model_path "outputs/chair_two"   --prompt "a photo of a chair with <asset0> and <asset1> and <asset6> and <asset7> on the street"   --output_path "outputs/chair_two_inference/asset0_asset1_asset6_asset7.jpg"
# python inference.py   --model_path "outputs/chair_two"   --prompt "a photo of a chair with <asset4> and <asset1> and <asset2> and <asset7> on the street"   --output_path "outputs/chair_two_inference/asset4_asset1_asset2_asset7.jpg"
# ### Swap 3 parts from img0 by 3 parts from img 1
# python inference.py   --model_path "outputs/chair_two"   --prompt "a photo of a chair with <asset4> and <asset5> and <asset6> and <asset3> on the street"   --output_path "outputs/chair_two_inference/asset4_asset5_asset6_asset3.jpg"
# python inference.py   --model_path "outputs/chair_two"   --prompt "a photo of a chair with <asset0> and <asset5> and <asset6> and <asset7> on the street"   --output_path "outputs/chair_two_inference/asset0_asset5_asset7_asset7.jpg"
# python inference.py   --model_path "outputs/chair_two"   --prompt "a photo of a chair with <asset4> and <asset1> and <asset6> and <asset7> on the street"   --output_path "outputs/chair_two_inference/asset4_asset1_asset6_asset7.jpg"
# python inference.py   --model_path "outputs/chair_two"   --prompt "a photo of a chair with <asset4> and <asset5> and <asset2> and <asset7> on the street"   --output_path "outputs/chair_two_inference/asset4_asset5_asset2_asset7.jpg"