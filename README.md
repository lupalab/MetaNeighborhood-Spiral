# MetaNeighborhood-Spiral
Visualize Meta-Neighborhood with 2d spiral data 

# Usage:
python main.py --update_lr=20 --num_updates=1 --init_with_training_data=False --num_slots=100

update_lr: inner loop learning rate.

num_updates: inner loop finetune steps.

init_with_training_data: whether initialize K and V with training data.

num_slots: number of slots in the memory.
