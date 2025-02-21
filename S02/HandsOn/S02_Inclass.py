import numpy as np
grid = np.array([
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [0, 0, 0, 2, 2, 2],
    [0, 0, 0, 2, 2, 2],
    [0, 0, 0, 2, 2, 2]
])
kernel = np.array([
    [1, 0],
    [0, 1]
])


def convolve(grid, kernel):
    H, W = grid.shape
    KH, KW = kernel.shape

    out_h = (H - KH)//4 + 1
    out_w = (W - KW)//4 + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = grid[i*4: i*4 + KH, j*4: j*4 + KW]
            output[i, j] = np.sum(region * kernel)

    return output


output = convolve(grid, kernel)
for row in output:
    print(row)
