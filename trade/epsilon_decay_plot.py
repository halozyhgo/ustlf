import numpy as np
import matplotlib.pyplot as plt

# 设置参数
epsilon_min = 0.01
epsilon_decay = 0.99
episodes = 1000

# 计算每个episode的epsilon值
epsilon_values = []
for episode in range(episodes):
    # epsilon = epsilon_min + (1.0 - epsilon_min) * (1 - np.exp(-epsilon_decay * episode / 100))
    epsilon = 1 - (1.0 - epsilon_min) * (1 - np.exp(-epsilon_decay * episode / 100))
    # epsilon = epsilon_min + (1.0 - epsilon_min) * np.exp(-epsilon_decay * episode)
    epsilon_values.append(epsilon)

# 绘制图像
plt.figure(figsize=(10, 5))
plt.plot(range(episodes), epsilon_values, label='Epsilon', color='red')  # 使用红色
plt.xlabel('Episode')
plt.ylabel('Epsilon Value')
plt.title('Epsilon Decay Over Episodes')
plt.grid()
plt.legend()
plt.savefig('epsilon_decay_plot.png')  # 保存图像
plt.show()