def euclidian_distance(point_a, point_b):
    return (sum([(point_a[i] - point_b[i])**2 for i in range(len(point_a))]))**0.5


def wcss(points):
    # Write code here
    # points = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    p = len(points)  # n = 3, number of points
    d = len(points[0])  # p = 3, number of dimensions

    if p == 1:
        return float(0)

    # Initialize certain variables
    centroid = []
    center = 0

    for i in range(0, d):  # i = 0, 1, 2: Number of dimensions
        for j in range(0, p):  # j = 0, 1, 2: Number of points/coordinates
            # Algorithm changes coordinate of concern before it changes dimensions, so coordinate loop is inside dimension loop
            center += points[j][i]  # center = 0. # 1st iteration: center = 1 + 4 + 7 = 12, 2nd iteration: center =  2 + 5 + 8 = 15, 3rd iteration: center = 3 + 6 + 9 = 18
        centroid.append(center/p)
        center = 0

    sum_of = 0

    for point in points:
        sum_of += euclidian_distance(point, centroid)
    
    return sum_of/p


if __name__ == "__main__":
    # Example usage
    points = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(wcss(points))  # Should be 3.464101615137755