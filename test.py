
image = [[0, 0, 0], [0, 0, 0]]
sr = 0
sc = 0
color = 0

origin = image[sr][sc]
m = len(image)
n = len(image[0])
already = [[0 for i in range(n)] for j in range(m)]


def fill(image, sr, sc, color, already):
    if already[sr][sc] == 1:
        return image, already
    already[sr][sc] = 1
    if image[sr][sc] == origin:
        image[sr][sc] = color
    else:
        return image, already
    if sr > 0:
        image, already = fill(image, sr - 1, sc, color, already)
    if sc > 0:
        image, already = fill(image, sr, sc - 1, color, already)
    if sr < m - 1:
        image, already = fill(image, sr + 1, sc, color, already)
    if sc < n - 1:
        image, already = fill(image, sr, sc + 1, color, already)
    return image, already

res, already = fill(image, sr, sc, color, already)
print(res)