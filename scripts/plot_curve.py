import math

def plot_real_curve():
    print("=" * 70)
    print("y^2 = x^3 + 4  over the REAL numbers (for intuition only)")
    print("=" * 70)
    width, height = 61, 31
    xmin, xmax = -3.0, 5.0
    ymin, ymax = -12.0, 12.0
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    for col in range(width):
        x = xmin + (xmax - xmin) * col / (width - 1)
        rhs = x**3 + 4
        if rhs < 0:
            continue
        y = math.sqrt(rhs)
        for yv in (y, -y):
            row = round((ymax - yv) / (ymax - ymin) * (height - 1))
            if 0 <= row < height:
                grid[row][col] = '*'

    axis_row = round((ymax - 0) / (ymax - ymin) * (height - 1))
    axis_col = round((0 - xmin) / (xmax - xmin) * (width - 1))
    for col in range(width):
        if grid[axis_row][col] == ' ':
            grid[axis_row][col] = '-'
    for row in range(height):
        if grid[row][axis_col] == ' ':
            grid[row][axis_col] = '|'
    grid[axis_row][axis_col] = '+'

    for row in grid:
        print(''.join(row))
    print("(smooth curve - continuous real numbers)\n")


def plot_toy_curve(p):
    print("=" * 70)
    print(f"y^2 = x^3 + 4  (mod {p})  <- what an EC actually looks like in crypto")
    print("=" * 70)
    points = []
    for x in range(p):
        rhs = (x**3 + 4) % p
        for y in range(p):
            if (y * y) % p == rhs:
                points.append((x, y))

    width, height = p, p // 2 + 1
    grid = [[' ' for _ in range(width)] for _ in range(height // 1 + 1)]
    # scale y down to keep terminal output compact (2 rows per y-unit horizontally is fine, but
    # let's just use half-height by printing every other row using block chars for density)
    rows = 30
    cols = min(p, 90)
    grid = [[' ' for _ in range(cols)] for _ in range(rows)]
    for x, y in points:
        col = x * (cols - 1) // (p - 1)
        row = rows - 1 - y * (rows - 1) // (p - 1)
        grid[row][col] = '*'

    for row in grid:
        print(''.join(row))
    print(f"({len(points)} points total, including the point at infinity, on F_{p})")
    print("Notice: no smooth shape at all - just a scattered cloud of points.")
    print("BLS12-381 works exactly like this, just with a 381-bit prime instead of 97,")
    print("so the 'cloud' has roughly 2^381 possible points instead of ~100.\n")


plot_real_curve()
plot_toy_curve(97)
