from .metropolis_hastings import probability_matrix, binomial_distribution, propose_state, metropolis_hastings
import numpy as np

def default_stafflines(shape, staff_lines):
    res = np.zeros(shape, np.uint8)

    for x0, x1, y0, y1 in staff_lines:
        for x in range(x0, x1 + 1):
            res[y0:y1, x:x+1] = 255
    
    return res

def staffline_interruptions(shape, staff_lines, num_states=4, c=0.9):
    res = np.zeros(shape, np.uint8)
    P = probability_matrix(num_states, c)

    for x0, x1, y0, y1 in staff_lines:
        gap = metropolis_hastings(binomial_distribution(num_states, 0.5), lambda x: propose_state(x, P), 0, x1 - x0 + 1)
        for x in range(x0, x1 + 1):
            if gap[x - x0]:
                res[y0:y1, x:x+1] = 255
    
    return res

def thickness_variations(shape, staff_lines, num_states=5, c=0.96):
    res = np.zeros(shape, np.uint8)
    P = probability_matrix(num_states, c)

    # determine state behaviour
    def transform(state, y0, y1):
        new_y0 = y0 + int(state / 2) - 1
        new_y1 = y1 + 1 - int((state + 1) / 2)
        return new_y0, new_y1

    for x0, x1, y0, y1 in staff_lines:
        thickness_variation = metropolis_hastings(binomial_distribution(num_states, 0.5), lambda x: propose_state(x, P), 0, x1 - x0 + 1)
        for x in range(x0, x1 + 1):
            cur_state = thickness_variation[x - x0]
            y0_new, y1_new = transform(cur_state, y0, y1)
            res[y0_new:y1_new, x] = 255
    
    return res

def y_variations(shape, staff_lines, num_states=5, c=0.96):
    res = np.zeros(shape, np.uint8)
    P = probability_matrix(num_states, c)

    # determine state behaviour
    def transform(state, y0, y1):
        new_y0 = y0 + state - 2
        new_y1 = y1 + state - 2
        return new_y0, new_y1

    for x0, x1, y0, y1 in staff_lines:
        y_variation = metropolis_hastings(binomial_distribution(num_states, 0.5), lambda x: propose_state(x, P), 0, x1 - x0 + 1)
        for x in range(x0, x1 + 1):
            cur_state = y_variation[x - x0]
            y0_new, y1_new = transform(cur_state, y0, y1)
            res[y0_new:y1_new, x] = 255

    return res