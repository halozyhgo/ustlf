import backtrader as bt
import pandas as pd
import datetime
import tushare as ts
import matplotlib.pyplot as plt

# 创建数据加载类
class TSPandasData(bt.feeds.PandasData):
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'vol'),
        ('openinterest', None),
    )

class MACrossStrategy(bt.Strategy):
    params = (
        ('fast_period', 10),       # 快速均线周期
        ('slow_period', 20),       # 慢速均线周期
        ('lookback', 126),         # 半年
        ('cross_threshold', 0.001), # 均线接近程度阈值
        ('slope_threshold', 0.001), # 斜率阈值
        ('slope_period', 3),       # 斜率计算周期
        ('stop_loss', 0.05),       # 止损比例，5%
        ('take_profit', 0.20),     # 止盈比例，20%
    )

    def __init__(self):
        # 计算均线
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow_period)
        
        # 计算历史高低价
        self.high_price = bt.indicators.Highest(self.data.high, period=self.params.lookback)
        self.low_price = bt.indicators.Lowest(self.data.low, period=self.params.lookback)
        
        # 交易状态
        self.order = None
        
        # 添加交易记录变量
        self.trade_price = None  # 记录开仓价格

    def calculate_slope(self, line, period):
        """计算均线斜率"""
        if len(line) < period:
            return 0
        
        y2, y1 = line[0], line[-period]
        return (y2 - y1) / period

    def should_close_position(self):
        """判断是否应该平仓"""
        # 计算快慢线当前和之前的斜率
        fast_slope_now = self.calculate_slope(self.fast_ma, self.params.slope_period)
        fast_slope_prev = self.calculate_slope(self.fast_ma.get(ago=-1, size=self.params.slope_period), self.params.slope_period)
        slow_slope_now = self.calculate_slope(self.slow_ma, self.params.slope_period)
        slow_slope_prev = self.calculate_slope(self.slow_ma.get(ago=-1, size=self.params.slope_period), self.params.slope_period)

        # 判断趋势变化
        if self.position.size > 0:  # 多头持仓
            return ((fast_slope_prev > 0 and fast_slope_now <= 0) or 
                   (slow_slope_prev > 0 and slow_slope_now <= 0))
        else:  # 空头持仓
            return ((fast_slope_prev < 0 and fast_slope_now >= 0) or 
                   (slow_slope_prev < 0 and slow_slope_now >= 0))

    def next(self):
        if self.order:
            return

        # 计算价格区间
        price_range = self.high_price[0] - self.low_price[0]
        upper_quarter = self.high_price[0] - price_range * 0.25
        lower_quarter = self.low_price[0] + price_range * 0.25

        # 检查趋势变化平仓条件
        if self.position and self.should_close_position():
            print(f'均线趋势改变，平仓')
            self.order = self.close()
            if self.position.size > 0:
                self.sell_up_arrow = True
            else:
                self.sell_down_arrow = True
            self.trade_price = None
            return

        # 检查均线水平条件（提前检查）
        if self.position and self.is_lines_horizontal():
            print(f'均线趋于水平，平仓，快线斜率: {self.calculate_slope(self.fast_ma, self.params.slope_period):.6f}, 慢线斜率: {self.calculate_slope(self.slow_ma, self.params.slope_period):.6f}')
            self.order = self.close()
            if self.position.size > 0:
                self.sell_up_arrow = True
            else:
                self.sell_down_arrow = True
            self.trade_price = None
            return

        # 检查止损和止盈条件
        if self.position and self.trade_price is not None:
            current_price = self.data.close[0]
            if self.position.size > 0:  # 多头持仓
                profit_loss_pct = (current_price - self.trade_price) / self.trade_price
                if profit_loss_pct <= -self.params.stop_loss:  # 亏损超过5%
                    print(f'触发止损: 多头浮亏 {-profit_loss_pct*100:.2f}%')
                    self.order = self.close()
                    self.sell_up_arrow = True
                    self.trade_price = None
                    return
                elif profit_loss_pct >= self.params.take_profit:  # 盈利超过20%
                    print(f'触发止盈: 多头盈利 {profit_loss_pct*100:.2f}%')
                    self.order = self.close()
                    self.sell_up_arrow = True
                    self.trade_price = None
                    return
            else:  # 空头持仓
                profit_loss_pct = (self.trade_price - current_price) / self.trade_price
                if profit_loss_pct <= -self.params.stop_loss:  # 亏损超过5%
                    print(f'触发止损: 空头浮亏 {-profit_loss_pct*100:.2f}%')
                    self.order = self.close()
                    self.sell_down_arrow = True
                    self.trade_price = None
                    return
                elif profit_loss_pct >= self.params.take_profit:  # 盈利超过20%
                    print(f'触发止盈: 空头盈利 {profit_loss_pct*100:.2f}%')
                    self.order = self.close()
                    self.sell_down_arrow = True
                    self.trade_price = None
                    return

        # 当前没有持仓
        if not self.position:
            # 即将形成死叉且价格在高位区间，做空
            if self.is_about_to_cross_down() and self.data.close[0] >= upper_quarter:
                self.order = self.sell()
                self.buy_down_arrow = True
                self.trade_price = self.data.close[0]  # 记录开仓价格
            # 即将形成金叉且价格在低位区间，做多
            elif self.is_about_to_cross_up() and self.data.close[0] <= lower_quarter:
                self.order = self.buy()
                self.buy_up_arrow = True
                self.trade_price = self.data.close[0]  # 记录开仓价格

        # 持有多头仓位
        elif self.position.size > 0:
            if self.is_about_to_cross_down():  # 即将形成死叉，平多
                self.order = self.close()
                self.sell_up_arrow = True
                self.trade_price = None

        # 持有空头仓位
        elif self.position.size < 0:
            if self.is_about_to_cross_up():  # 即将形成金叉，平空
                self.order = self.close()
                self.sell_down_arrow = True
                self.trade_price = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                print(f'买入执行价格: {order.executed.price:.2f}')
            elif order.issell():
                print(f'卖出执行价格: {order.executed.price:.2f}')
            
            # 如果是开仓，记录交易价格
            if not self.position:
                self.trade_price = order.executed.price

        self.order = None

    def is_about_to_cross_up(self):
        """判断是否即将形成金叉"""
        fast_slope = self.calculate_slope(self.fast_ma, self.params.slope_period)
        slow_slope = self.calculate_slope(self.slow_ma, self.params.slope_period)
        
        # 放宽条件
        return (self.fast_ma[-1] < self.slow_ma[-1] and 
                self.fast_ma[0] < self.slow_ma[0] and
                fast_slope > 0 and  # 只要快线向上
                (self.slow_ma[0] - self.fast_ma[0]) / self.slow_ma[0] < self.params.cross_threshold)

    def is_about_to_cross_down(self):
        """判断是否即将形成死叉"""
        fast_slope = self.calculate_slope(self.fast_ma, self.params.slope_period)
        slow_slope = self.calculate_slope(self.slow_ma, self.params.slope_period)
        
        # 放宽条件
        return (self.fast_ma[-1] > self.slow_ma[-1] and 
                self.fast_ma[0] > self.slow_ma[0] and
                fast_slope < 0 and  # 只要快线向下
                (self.fast_ma[0] - self.slow_ma[0]) / self.slow_ma[0] < self.params.cross_threshold)

    def is_lines_horizontal(self):
        """判断均线是否趋于水平"""
        fast_slope = self.calculate_slope(self.fast_ma, self.params.slope_period)
        slow_slope = self.calculate_slope(self.slow_ma, self.params.slope_period)
        
        # 放宽判断条件：任一均线接近水平即可
        return (abs(fast_slope) < self.params.slope_threshold or 
                abs(slow_slope) < self.params.slope_threshold)

def run_backtest():
    cerebro = bt.Cerebro()
    
    try:
        # 使用tushare获取数据
        token = '7dd0a9658076ce20bebdd03d5ac66a8485546019267214fb0a189698'
        ts.set_token(token)
        pro = ts.pro_api()
        
        # 获取3年数据（2021-2023）
        df = pro.fut_daily(
            ts_code='M.DCE', 
            start_date='20210101',  # 改为2021年开始
            end_date='20231231',    # 到2023年底
            fields='trade_date,open,high,low,close,vol'
        )
        
        if df is None or df.empty:
            print("未获取到数据，请检查token和网络连接")
            return
            
        # 数据处理
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        df.sort_index(inplace=True)
            
        print("\n====== 数据信息 ======")
        print(f"数据时间范围: {df.index.min()} 到 {df.index.max()}")
        print(f"数据条数: {len(df)}")
        print("\n数据样例:")
        print(df.head())
        
    except Exception as e:
        print("获取数据时出错:", str(e))
        return
    
    # 创建数据源
    data = TSPandasData(dataname=df)
    cerebro.adddata(data)
    
    # 设置初始资金和手续费
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.0003)
    
    # 添加策略
    cerebro.addstrategy(MACrossStrategy)
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    # 运行回测
    print('初始资金: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    strat = results[0]
    
    # 打印回测结果
    print('\n====== 回测结果 ======')
    print('最终资金: %.2f' % cerebro.broker.getvalue())
    print('夏普比率:', strat.analyzers.sharpe.get_analysis()['sharperatio'])
    print('最大回撤: %.2f%%' % strat.analyzers.drawdown.get_analysis()['max']['drawdown'])
    print('年化收益率: %.2f%%' % (strat.analyzers.returns.get_analysis()['rnorm100']))
    
    # 绘制结果
    cerebro.plot(style='candlestick',
                barup='red', bardown='green',  # 设置K线颜色
                plotdist=0.1,
                volume=True,
                fmt_x_ticks='%Y%m%d',
                buysell=True,  # 显示买卖点
                buysell_markers=('*', '*', 's', 's'),  # 设置买卖标记的形状：买入红星、卖出绿星、做空红方块、平空绿方块
                buysell_markersize=(8, 8, 8, 8))  # 设置标记大小

if __name__ == '__main__':
    run_backtest() 