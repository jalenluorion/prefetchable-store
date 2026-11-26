from intervals import merge

cases = [
    ([[1, 3], [2, 6], [8, 10]], [[1, 6], [8, 10]]),
    ([[1, 4], [4, 5]], [[1, 5]]),
    ([[1, 10], [2, 3], [4, 5]], [[1, 10]]),
]
for given, want in cases:
    got = merge([list(i) for i in given])
    if got != want:
        raise SystemExit(f"FAIL merge({given}) = {got}, expected {want}")
print("all checks pass")
