import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from IPython.display import HTML
import time

# 定义参数
radius = 1
elasticity = 0.9  # 碰撞后的弹性系数
num_balls = 5  # 小球的数量
wall_x = [0, 100]  # x方向墙壁的范围
wall_y = [0, 100]  # y方向墙壁的范围
wall_z = [0, 100]  # z方向墙壁的范围

# 获取系统启动时的时间戳
boot_time = time.monotonic()

# 随时间变化的重力场，基于时间 t 来计算三维重力
def calculate_gravity(t):
    # 重力在三个轴上的方向随时间变化
    angle_x = np.pi * t / 5
    angle_y = np.pi * t / 3
    angle_z = np.pi * t / 7

    gravity_x = 500 * np.sin(angle_x)  # x方向重力场随时间变化
    gravity_y = 500 * np.sin(angle_y)  # y方向重力场随时间变化
    gravity_z = 9810 * np.sin(angle_z)  # z方向重力场随时间变化

    return np.array([gravity_x, gravity_y, gravity_z])

# 随机初始化小球的位置和速度
np.random.seed(0)  # 固定随机种子，确保结果可复现
initial_pos = np.random.uniform(low=[radius, radius, radius], high=[wall_x[1] - radius, wall_y[1] - radius, wall_z[1] - radius], size=(num_balls, 3))
initial_velocity = np.random.uniform(low=-2, high=2, size=(num_balls, 3))  # 初始速度较小

# 初始化小球位置和速度
positions = np.array(initial_pos)
velocities = np.array(initial_velocity)

# 增加调试输出，检查小球初始位置
print("Initial positions:", positions)

# 创建图形和三维轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(wall_x[0], wall_x[1])
ax.set_ylim(wall_y[0], wall_y[1])
ax.set_zlim(wall_z[0], wall_z[1])

# 确保小球的 markersize 够大
balls, = ax.plot([], [], [], 'o', markersize=radius * 15, color='blue')

# 碰撞检测与处理
def handle_collision(i, j, positions, velocities):
    distance = np.linalg.norm(positions[i] - positions[j])
    if distance < 2 * radius:  # 如果小球之间发生碰撞
        normal = (positions[j] - positions[i]) / distance
        relative_velocity = velocities[i] - velocities[j]
        velocity_along_normal = np.dot(relative_velocity, normal)

        if velocity_along_normal > 0:
            return  # 小球正分离，不处理

        restitution = elasticity  # 弹性碰撞系数
        impulse = (1 + restitution) * velocity_along_normal / 2
        velocities[i] -= impulse * normal
        velocities[j] += impulse * normal

# 更新函数时，确认小球位置在数据中被正确传递
def update(frame):
    global positions, velocities

    # 获取当前时间 t
    current_time = time.monotonic()
    uptime_seconds = current_time - boot_time
    t = uptime_seconds

    # 动态计算当前的重力方向
    gravity = calculate_gravity(t)

    # 更新速度和位置
    velocities += gravity * 0.01  # 应用动态重力
    positions += velocities

    # 碰撞检测与反弹
    for i in range(num_balls):
        # x方向碰撞检测
        if positions[i, 0] - radius < wall_x[0]:
            velocities[i, 0] = -velocities[i, 0] * elasticity
            positions[i, 0] = wall_x[0] + radius  # 确保小球不穿透墙壁
        elif positions[i, 0] + radius > wall_x[1]:
            velocities[i, 0] = -velocities[i, 0] * elasticity
            positions[i, 0] = wall_x[1] - radius  # 确保小球不穿透墙壁

        # y方向碰撞检测
        if positions[i, 1] - radius < wall_y[0]:
            velocities[i, 1] = -velocities[i, 1] * elasticity
            positions[i, 1] = wall_y[0] + radius  # 确保小球不穿透墙壁
        elif positions[i, 1] + radius > wall_y[1]:
            velocities[i, 1] = -velocities[i, 1] * elasticity
            positions[i, 1] = wall_y[1] - radius  # 确保小球不穿透墙壁

        # z方向碰撞检测
        if positions[i, 2] - radius < wall_z[0]:
            velocities[i, 2] = -velocities[i, 2] * elasticity
            positions[i, 2] = wall_z[0] + radius  # 确保小球不穿透墙壁
        elif positions[i, 2] + radius > wall_z[1]:
            velocities[i, 2] = -velocities[i, 2] * elasticity
            positions[i, 2] = wall_z[1] - radius  # 确保小球不穿透墙壁

        # 碰撞小球
        for j in range(i + 1, num_balls):
            handle_collision(i, j, positions, velocities)

    # 更新小球位置到 3D 图形
    balls.set_data(positions[:, 0], positions[:, 1])
    balls.set_3d_properties(positions[:, 2])
    return balls,

# 创建动画
ani = animation.FuncAnimation(
    fig, update, frames=200, init_func=None, blit=True, interval=50
)

# 显示动画
HTML(ani.to_jshtml())

# 创建动画并保存为 mp4 文件
ani.save('collision_simulation_3d.mp4', writer='ffmpeg')
