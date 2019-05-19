# MetaNeighborhood-Spiral
Visualize Meta-Neighborhood with 2d spiral data 

# Usage:

meta model:

python main.py

vanillla model:

python main.py --vanilla=True

Figures illustrating the movement of keys will be saved in a folder in the root dir.

![](random.gif)

update_lr: inner loop learning rate.

num_updates: inner loop finetune steps.

init_with_training_data: whether initialize K and V with training data.

num_slots: number of slots in the memory.

