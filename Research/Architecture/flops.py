def calculate_LE(K, K_max, SIZE, FLOPs):
    LE = (K / K_max) * sum(size + flops for size, flops in zip(SIZE, FLOPs))
    return LE


# Usage
K = 20
K_max = 100
# Assuming SIZE and FLOPs to be lists with values
SIZE = [5, 10, 20, 15]
FLOPs = [15, 20, 30, 40]
LE = calculate_LE(K, K_max, SIZE, FLOPs)
print(LE)
