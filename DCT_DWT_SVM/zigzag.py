import numpy as np


def zigzag(shape):
    ########################
    # (ONLY WORKS FOR SQUARE SHAPES, FURTHER CHANGES MAY BE USEFUL AND IS SUBJECT TO LATER DEVELOPMENT)
    # perform the same zig-zag assigment that's used in 71
    #
    # shape = tuple (x,x) for this paper (10,10) (can take and produce any square shape)
    # output = a list of appropriate indexes to assign from a flattened matrix
    #           (i.e. output = [0, 1, 10, 20 ...] for a 10x10 zigzag pattern. Given a 10x10 arbitrary matrix  A that's
    #           flattened with np.flatten(), and the zigzag matrix will be 1d B = [A[0], A[1], A[10], A[20] ...]
    #           The number of zigzag features chosen is directly controllable so B[0:100] will return 100 features
    #           in a zigzag order. Please note that the output is simply the appropriate index, and this function does
    #           not directly create B for reasons of optimization
    #########################

    ###########
    # preamble
    ###########
    x = np.zeros((shape[0] * shape[1], 1))
    y = np.zeros((shape[0] * shape[1], 1))
    t = 0
    cond_x = True
    cond_y = True
    cond = True
    x_t = 0
    y_t = 0

    ############
    # Build zig-zag pattern
    ############

    while cond:
        # condition verifications
        if x[t] == shape[0] - 1 and cond_x:
            x_t = t
            cond_x = False
        elif y[t] == shape[1] - 1 and cond_y:
            y_t = t
            cond_y = False
        elif not cond_x and not cond_y:
            cond = False

        # reassignments, t is ++ inside because of nested while loops
        if t == 0:
            y[t + 1] += 1
            t += 1
        elif x[t] == 0:
            if x[t - 1] == x[t] and y[t - 1] == y[t] - 1:
                while not y[t] == 0:
                    x[t + 1] = x[t] + 1
                    y[t + 1] = y[t] - 1
                    t += 1
            else:
                y[t + 1] = y[t] + 1
                t += 1
        elif y[t] == 0:
            if y[t - 1] == y[t] and x[t - 1] == x[t] - 1:
                while not x[t] == 0:
                    x[t + 1] = x[t] - 1
                    y[t + 1] = y[t] + 1
                    t += 1
            else:
                x[t + 1] = x[t] + 1
                t += 1

    # Due to symmetry, the following mirror can be performed
    x_r = (shape[0] - 1) - x[0:y_t]
    y_r = (shape[1] - 1) - y[0:y_t]
    x[x_t + 1:shape[0] * shape[1]] = np.flip(x_r)
    y[x_t + 1:shape[0] * shape[1]] = np.flip(y_r)

    ###########
    # Create index matrix and return
    ###########

    # fill a linear matrix with the corresponding index in original flattened matrix corresponding to position
    result = np.zeros((1, int(shape[0]) * int(shape[1])))
    for i in range(shape[0] * shape[1]):
        result[0, i] = int(x[i])*shape[0] + int(y[i])

    return result
