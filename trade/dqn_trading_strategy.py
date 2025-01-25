import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tushare as ts
from collections import deque
import random
import matplotlib.pyplot as plt
import mplfinance as mpf
import os
from datetime import datetime

class DQNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQNNetwork, self).__init__()
        self.lookback = 30  # 时间序列长度
        
        # 使用1D卷积处理时间序列数据
        self.conv1 = nn.Conv1d(2, 16, kernel_size=3)  # 2个通道(开盘价和收盘价)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        
        # 计算卷积后的特征维度
        conv_out_size = 32 * 6  # 32个通道，6个特征
        
        # 全连接层
        self.fc1 = nn.Linear(conv_out_size + 5, hidden_size)  # +5是其他特征的维度
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 分离时间序列数据和其他特征
        batch_size = x.size(0)
        time_series = x[:, :-5].view(batch_size, 2, -1)  # 重塑为[batch, 2, 30]
        other_features = x[:, -5:]
        
        # 处理时间序列
        x = self.relu(self.conv1(time_series))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(batch_size, -1)  # 展平
        
        # 合并其他特征
        x = torch.cat([x, other_features], dim=1)
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class TradingEnvironment:
    def __init__(self, data, lookback=30):
        self.data = data
        self.lookback = lookback
        self.commission_rate = 0.0003  # 手续费率 0.03%
        self.min_holding_period = 5    # 最小持仓周期
        self.holding_time = 0          # 当前持仓时间
        self.reset()
        
    def reset(self):
        self.current_step = self.lookback
        self.position = 0  # -1: 空仓, 0: 无仓位, 1: 多仓
        self.cash = 100000
        self.holdings = 0
        self.total_value = self.cash
        self.trades = []  # 记录交易历史
        self.holding_time = 0  # 重置持仓时间
        self.last_trade_price = None  # 上次交易价格
        return self._get_state()
    
    def _get_state(self):
        # 获取历史价格数据
        prices = self.data['close'].values[self.current_step-self.lookback:self.current_step]
        opens = self.data['open'].values[self.current_step-self.lookback:self.current_step]
        
        # 对价格数据进行归一化，使用最近一个价格作为基准
        norm_prices = prices / prices[-1]  # 归一化收盘价序列
        norm_opens = opens / prices[-1]    # 归一化开盘价序列
        
        # 计算技术指标
        returns = np.diff(prices) / prices[:-1]
        ma10 = np.mean(prices[-10:])
        ma20 = np.mean(prices)
        std = np.std(returns)
        
        # 组合所有状态要素
        state = np.concatenate([
            norm_prices,           # 30个归一化的收盘价
            norm_opens,           # 30个归一化的开盘价
            np.array([
                prices[-1]/prices[-2] - 1,  # 最新收益率
                prices[-1]/ma10 - 1,        # 价格相对10日均线
                prices[-1]/ma20 - 1,        # 价格相对20日均线
                std,                        # 波动率
                self.position,              # 当前持仓状态
            ])
        ])
        return state
    
    def step(self, action):
        # 确保动作在有效范围内
        if action < 0 or action >= 3:
            raise ValueError(f"Invalid action: {action}")

        current_price = self.data['close'].values[self.current_step]
        next_price = self.data['close'].values[self.current_step + 1]
        
        # 计算基础收益率
        price_change = (next_price - current_price) / current_price
        
        # 初始化奖励
        reward = 0
        old_position = self.position
        
        # 更新持仓时间
        if self.position != 0:
            self.holding_time += 1
        
        # 执行交易
        if action == 0:  # 卖出/空仓
            if self.position != -1:  # 新开空仓或平多仓
                if self.holding_time < self.min_holding_period and self.position != 0:
                    reward = -0.1  # 惩罚频繁交易
                else:
                    reward = -price_change  # 直接使用价格变化作为奖励
                    self.total_value *= (1 - self.commission_rate)  # 扣除手续费
                    self.position = -1
                    self.holding_time = 0
                    self.last_trade_price = current_price
                
        elif action == 1:  # 持有
            if self.position == 1:  # 持有多仓
                reward = price_change
            elif self.position == -1:  # 持有空仓
                reward = -price_change
        
        elif action == 2:  # 买入
            if self.position != 1:  # 新开多仓或平空仓
                if self.holding_time < self.min_holding_period and self.position != 0:
                    reward = -0.1  # 惩罚频繁交易
                else:
                    reward = price_change  # 直接使用价格变化作为奖励
                    self.total_value *= (1 - self.commission_rate)  # 扣除手续费
                    self.position = 1
                    self.holding_time = 0
                    self.last_trade_price = current_price
                
        # 更新资金
        self.total_value *= (1 + reward)
        
        # 记录交易
        if action != 1 and old_position != self.position:  # 只在实际发生交易时记录
            self.trades.append({
                'date': self.data.index[self.current_step],
                'action': 'buy' if action == 2 else 'sell',
                'price': current_price,
                'value': self.total_value
            })
        
        # 移动到下一步
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        return self._get_state(), reward, done

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # 折扣因子
        self.epsilon = 1.0   # 初始探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.99  # 衰减因子
        self.learning_rate = 0.0005  # 尝试不同的学习率
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DQNNetwork(state_size, 64, action_size).to(self.device)
        self.target_model = DQNNetwork(state_size, 64, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def update_epsilon(self, episode):
        """更新epsilon值，使其呈凹形变化"""
        self.epsilon = 1 - (1.0 - self.epsilon_min) * (1 - np.exp(-self.epsilon_decay * episode / 100))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)  # 确保返回的动作在范围内
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            act_values = self.model(state)
            return torch.argmax(act_values[0]).item()  # 返回最大Q值对应的动作
            
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 当前Q值
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # 下一步最大Q值
        next_q = self.target_model(next_states).detach().max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 计算损失并更新
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

def plot_trading_actions(df, actions, episode, env, output_dir):
    """绘制交易行为图"""
    # 准备数据
    plot_data = df[env.lookback:env.current_step+1].copy()
    plot_data = plot_data.rename(columns={'vol': 'volume'})
    
    # 创建买卖点标记和资金记录
    buy_signals = []
    sell_signals = []
    positions = []
    current_pos = 0
    equity_marks = []  # 记录资金标记点
    
    # 记录初始资金
    equity_marks.append({
        'date': plot_data.index[0],
        'value': 100000,
        'position': plot_data.loc[plot_data.index[0], 'low'] * 0.97
    })
    
    # 按月记录资金
    current_month = plot_data.index[0].month
    current_value = 100000
    
    for i, action in enumerate(actions):
        current_date = plot_data.index[i]
        
        # 更新持仓状态和交易信号
        if action == 2 and current_pos != 1:
            buy_signals.append(current_date)
            current_pos = 1
        elif action == 0 and current_pos != -1:
            sell_signals.append(current_date)
            current_pos = -1
        positions.append(current_pos)
        
        # 更新当前资金
        if i > 0:
            price_change = (plot_data['close'].iloc[i] - plot_data['close'].iloc[i-1]) / plot_data['close'].iloc[i-1]
            if current_pos == 1:
                current_value *= (1 + price_change)
            elif current_pos == -1:
                current_value *= (1 - price_change)
        
        # 每月记录一次资金
        if current_date.month != current_month:
            equity_marks.append({
                'date': current_date,
                'value': current_value,
                'position': plot_data.loc[current_date, 'low'] * 0.97
            })
            current_month = current_date.month
    
    # 记录最终资金
    equity_marks.append({
        'date': plot_data.index[-1],
        'value': current_value,
        'position': plot_data.loc[plot_data.index[-1], 'low'] * 0.97
    })
    
    # 设置绘图风格和均线
    mc = mpf.make_marketcolors(up='red',down='green',
                              edge='inherit',
                              wick='inherit',
                              volume='in',
                              inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc)
    
    # 添加均线
    exp10 = plot_data['close'].ewm(span=10, adjust=False).mean()
    exp20 = plot_data['close'].ewm(span=20, adjust=False).mean()
    addplot = [
        mpf.make_addplot(exp10, color='blue', width=0.8),
        mpf.make_addplot(exp20, color='orange', width=0.8)
    ]
    
    # 绘制图形
    fig, axlist = mpf.plot(plot_data, 
                          type='candle',
                          style=s,
                          volume=True,
                          figsize=(15, 10),
                          title=f'\nTrading Actions - Episode {episode+1}\n'
                                f'Initial: ¥100,000  Final: ¥{env.total_value:.2f}\n'
                                f'Return: {((env.total_value/100000 - 1)*100):.2f}%',
                          returnfig=True,
                          addplot=addplot,
                          panel_ratios=(3,1))
    
    # 添加买卖点标记
    ax = axlist[0]
    for buy_point in buy_signals:
        ax.plot(plot_data.index.get_loc(buy_point), 
                plot_data.loc[buy_point, 'low'] * 0.99,
                '^', color='red', markersize=10, label='Buy' if buy_point == buy_signals[0] else "")
    
    for sell_point in sell_signals:
        ax.plot(plot_data.index.get_loc(sell_point),
                plot_data.loc[sell_point, 'high'] * 1.01,
                'v', color='green', markersize=10, label='Sell' if sell_point == sell_signals[0] else "")
    
    # 添加资金标记
    for mark in equity_marks:
        ax.text(plot_data.index.get_loc(mark['date']), 
                mark['position'],
                f"¥{mark['value']:,.0f}",
                color='black',
                fontsize=8,
                rotation=90)
    
    # 添加图例
    ax.legend()
    
    # 保存图片
    plt.savefig(os.path.join(output_dir, f'trading_episode_{episode+1}.png'), dpi=200, bbox_inches='tight')
    plt.close()

def train_model():
    # 创建时间戳文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"trading_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 获取数据
    token = '7dd0a9658076ce20bebdd03d5ac66a8485546019267214fb0a189698'
    ts.set_token(token)
    pro = ts.pro_api()
    
    df = pro.fut_daily(
        ts_code='M.DCE', 
        start_date='20210101',
        end_date='20231231',
        fields='trade_date,open,high,low,close,vol'
    )
    
    # 打印数据以检查
    print("获取的数据：")
    print(df.head())  # 打印前几行数据
    print("数据列名：", df.columns)  # 打印列名
    
    # 检查数据是否为空
    if df is None or df.empty:
        print("未获取到数据，请检查token和网络连接")
        return
    
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    df.sort_index(inplace=True)
    
    # 创建环境和智能体
    env = TradingEnvironment(df)
    state_size = 65  # 30(收盘价) + 30(开盘价) + 5(其他特征)
    action_size = 3  # 动作空间大小（卖出/空仓，持有，买入）
    agent = DQNAgent(state_size, action_size)
    
    # 训练参数
    episodes = 1000  # 增加训练轮数
    batch_size = 32
    
    # 记录训练历史
    history = []
    
    # 开始训练
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        actions_count = {0: 0, 1: 0, 2: 0}
        actions_history = []  # 记录每步的动作
        
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            actions_count[action] += 1
            actions_history.append(action)  # 记录动作
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            if done:
                agent.update_target_model()
                agent.update_epsilon(episode)  # 更新epsilon
                history.append(total_reward)
                print(f"Episode: {episode + 1}/{episodes}, "
                      f"Total Reward: {total_reward:.2f}, "
                      f"Epsilon: {agent.epsilon:.3f}, "
                      f"Actions: Buy={actions_count[2]}, "
                      f"Hold={actions_count[1]}, "
                      f"Sell={actions_count[0]}")
                
                # 绘制并保存交易图
                plot_trading_actions(df, actions_history, episode, env, output_dir)
                break
        
        history.append(total_reward)
    
    # 打印奖励变化趋势图
    plt.figure(figsize=(10, 5))
    plt.plot(history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Reward Over Episodes')
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'reward_trend.png'))  # 保存奖励变化趋势图
    plt.show()
    
    # 打印训练统计信息
    print("\n====== 训练统计 ======")
    print(f"平均奖励: {np.mean(history):.2f}")
    print(f"最大奖励: {np.max(history):.2f}")
    print(f"最小奖励: {np.min(history):.2f}")
    
    return agent, env, history

def test_model(agent, env):
    state = env.reset()
    total_reward = 0
    actions_taken = []
    
    while True:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        actions_taken.append(action)
        state = next_state
        
        if done:
            break
    
    print(f"Test Performance - Total Reward: {total_reward:.2f}")
    return actions_taken

if __name__ == "__main__":
    # 训练模型
    trained_agent, env, history = train_model()
    
    # 测试模型
    actions = test_model(trained_agent, env)
    
    # 保存模型
    torch.save(trained_agent.model.state_dict(), 'dqn_trading_model.pth') 