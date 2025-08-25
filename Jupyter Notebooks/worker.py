import tensorflow as tf

def generate_candidates_for_beam(beam_item, regions, num_regions):
    """
    Worker function that takes one beam path and generates all possible next candidate steps.
    This is defined in a separate file to be importable by spawned processes.
    """
    i, score, seq, img_state = beam_item
    generated_candidates = []
    
    # Define constants needed for the perturbation
    GRID_SIZE = 4
    REGION_HEIGHT = 224 // GRID_SIZE
    REGION_WIDTH = 224 // GRID_SIZE

    remaining_regions = [r for r in range(num_regions) if r not in seq]
    for region_idx in remaining_regions:
        y1, y2, x1, x2 = regions[region_idx]
        
        # This TensorFlow operation will run on a CPU core in the worker process
        noise = tf.random.uniform(shape=(REGION_HEIGHT, REGION_WIDTH, 3), minval=-128, maxval=128)
        indices = [[y, x, c] for y in range(y1, y2) for x in range(x1, x2) for c in range(3)]
        perturbed_image = tf.tensor_scatter_nd_update(img_state, indices, tf.reshape(noise, [-1]))

        # The map_info dictionary stores metadata to reconstruct the path later
        map_info = {
            'image_idx': i,
            'prev_score': score,
            'sequence': seq + [region_idx]
        }
        generated_candidates.append((perturbed_image, map_info))
        
    return generated_candidates