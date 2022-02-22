import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_triangles():
    m = ((50-10*(5**(1/2)))**(1/2))/10
    n = ((50+10*(5**(1/2)))**(1/2))/10

    # print(m, n)

    viewpoints = [[m, 0, n], [-m, 0, n], [m, 0, -n], [-m, 0, -n],
                  [0, n, m], [0, n, -m], [0, -n, m], [0, -n, -m],
                  [n, m, 0], [n, -m, 0], [-n, m, 0], [-n, -m, 0]]

    viewpoints = np.asarray(viewpoints)

    indices = []
    triangle_indices = set()

    for i in range(len(viewpoints)):
        for j in range(i+1, len(viewpoints)):
            print(np.linalg.norm(viewpoints[i]-viewpoints[j]))
            if round(np.linalg.norm(viewpoints[i]-viewpoints[j]), 1) == 1.1:
                # print(i, j, np.linalg.norm(viewpoints[i]-viewpoints[j]))
                indices.append([i, j])

    print(len(indices))    # 30条棱
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            set1 = set(indices[i])
            set2 = set(indices[j])
            if set1 & set2:    # 如果有交集
                ssd = set1 ^ set2    # 对称差集 "symmetric set difference"
                if list(ssd) in indices:
                    print(set1 | set2 | ssd)    # 打印并集
                    triangle_indices.add(tuple(sorted(list(set1 | set2 | ssd))))

    triangles = []    # 一共有20个面
    for t_i in triangle_indices:
        total_points.append(viewpoints[t_i[0]])
        total_points.append(viewpoints[t_i[1]])
        total_points.append(viewpoints[t_i[2]])
        # print(viewpoints[t_i[0]], viewpoints[t_i[1]], viewpoints[t_i[2]])
        triangles.append(viewpoints[np.array(t_i)])
    return triangles


def sample_points(data, accum=2):
    global total_data, total_points
    new_data = []
    for triangle in data:
        # triangle中存着三角形的三个顶点的坐标
        # 求三个顶点的三个中点
        center_point1 = np.array((triangle[0]+triangle[1])/2)
        center_point2 = np.array((triangle[0]+triangle[2])/2)
        center_point3 = np.array((triangle[1]+triangle[2])/2)

        center_point1 = center_point1/np.linalg.norm(center_point1)
        center_point2 = center_point2 / np.linalg.norm(center_point2)
        center_point3 = center_point3 / np.linalg.norm(center_point3)
        total_points.append(center_point1)
        total_points.append(center_point2)
        total_points.append(center_point3)
        # total_points += list(triangle)

        new_data.append([triangle[0], center_point1, center_point2])
        new_data.append([triangle[1], center_point1, center_point3])
        new_data.append([triangle[2], center_point2, center_point3])
        new_data.append([center_point1, center_point2, center_point3])

    print(new_data)
    total_data += new_data
    if accum == 0:
        return
    else:
        sample_points(new_data, accum-1)


total_data = []
total_points = []

triangles = get_triangles()    # 20个面
sample_points(triangles)

fig = plt.figure()
ax = Axes3D(fig=fig)

print(len(total_points))
color = []
total_points = np.asarray(total_points)
for view in total_points:
    color.append('r')

ax.scatter(total_points[:, 0], total_points[:, 1], total_points[:, 2], color=color, s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()