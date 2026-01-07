"""
多因子选股模型——深度融合机器学习算法：
"""
from atrader import *  
import pandas as pd
import numpy as np
from sklearn.svm import SVR # SVM回归
from sklearn.model_selection import GridSearchCV  # 网格搜索
from sklearn.model_selection import learning_curve # 学习曲线
from sklearn.linear_model import LinearRegression  # 线性回归
from sklearn.ensemble import RandomForestRegressor # 随机森林
import math

  
def init(context: Context):
    # 设置回测初始资金和股票交易费用
    set_backtest(initial_cash=initial_cash, future_cost_fee=1.0, stock_cost_fee=2.5,
                 margin_rate=1.0, slide_price=0, price_loc=1, deal_type=0, limit_type=0)
    # 注册因子数据
    reg_factor(factor_list)
    context.Tlen = len(context.target_list)
    target_all = pd.Series(np.char.array(context.target_list).lower())
    context.code_list = pd.DataFrame(target_all, columns=['code'])
    context.code_list.reset_index(inplace=True)
    context.code_list.rename(columns={'index': 'target_idx'}, inplace=True)
    context.code_list.set_index('target_idx', inplace=True)
    # 获取交易日历
    context.fre_date = get_trade_date_list(begin_date, end_date, _period=freq[0], begin_or_end=freq[1])
    # 初始化一个空字典来存储每个交易日的股票代码列表
    context.cons = {}
    # 遍历交易日历，获取每个交易日的股票代码列表
    for date in context.fre_date:
        context.cons[date] = get_code_list(universe, date)['code'].tolist()

def on_data(context):   
    # 获取当前日期并格式化,打印当前日期
    current_date = context.now.strftime('%Y-%m-%d')
    print(current_date)
    # 判断当前日期是否在调仓日期列表中
    if current_date in context.fre_date:  
        # 获取当前日期的股票代码列表
        current_code = context.cons[current_date] 

        # 获取因子值(市值和营业收入的增长率)
        factor = get_reg_factor(reg_idx=context.reg_factor[0],
                                    target_indices=(), length=1, df=True, sort_by='date')
        if len(factor) == 0:
            return
        factor_df = pd.DataFrame(index = current_code)  
        for i in range(len(factor_list)):
            factor_data = factor.groupby('factor').get_group(factor_list[i])  
            # 如果因子的方向为负，则对因子值取倒数
            if direction[i] == -1:
                factor_data['value'] = 1/factor_data['value']  
            factor_data.set_index('target_idx',inplace=True)  
            factor_data = pd.merge(context.code_list,factor_data,left_index=True,right_index=True,how='inner')
            factor_data = factor_data.set_index('code').reindex(current_code) 
            # 获取因子数据的值
            factor_data = factor_data['value'] 
            if factor_processing[i]['winsorize'][0] == True:
                # 使用 mad_winsorize_series 函数进行极值处理
                factor_data = mad_winsorize_series(factor_data,factor_processing[i]['winsorize'][1])
            # 如果需要对因子数据进行标准化处理
            # if factor_processing[i]['standardize'] == True:
            #     # 使用 standardize_series 函数进行标准化处理
            #     factor_data = standardize_series(factor_data)  
            factor_data.name = factor_list[i]
            factor_df = pd.concat([factor_df, factor_data], axis=1)
            factor_df.rename(columns={factor_list[0]: 'log_mcap'}, inplace=True)
            factor_df.rename(columns={factor_list[1]: 'g'}, inplace=True)

        # 获取财报数据
        order_book_ids = current_code
        date = current_date
        result_data = get_recent_quarter_data_all(order_book_ids, date)
        if len(result_data) == 0:
            return
        result = result_data[['code', 'quarter', 'publish_date','total_assets',
                'total_liabilities','net_profit','impairment_intangible_assets']].copy()
        result.loc[:, 'log_NC'] = result['total_assets'] - result['total_liabilities']
        result.loc[:, 'LEV'] = result['total_assets'] / result['total_liabilities']
        result = result.rename(columns={'net_profit': 'NI_p'})
        result.loc[:, 'NI_n'] = result['NI_p'] + 1
        result = result.rename(columns={'impairment_intangible_assets': 'log_RD'})
        result = result.drop(['quarter', 'publish_date','total_assets', 'total_liabilities'], axis=1)
        result = result.set_index('code').reindex(current_code)

        # 合并财报数据与因子数据
        merged_result = pd.merge(factor_df, result, left_index=True, right_index=True, how='inner')
        # 定义想要的列顺序
        columns_order = ['log_mcap', 'log_NC', 'LEV', 'NI_p', 'NI_n', 'g', 'log_RD']
        df = merged_result[columns_order].copy()
        # 对指定列进行极值处理并覆盖原来的值
        for col in X_list:
            if X_processing[col]['winsorize'][0] == True:
                df[col] = mad_winsorize_series(df[col], X_processing[col]['winsorize'][1])
        # 对指定列进行标准化处理并覆盖原来的值
        #     if X_processing[col]['standardize'] == True:
        #         df[col] = standardize_series(df[col])
        # 指定要进行对数转换的列名列表
        columns_to_transform = ['log_mcap','log_NC','NI_p','NI_n','log_RD']
        # 循环对每列进行操作
        for col in columns_to_transform:
            df[col] = df[col].apply(lambda x: np.log(x) if pd.notnull(x) else x)
        df = df.fillna(0)
        df[df > 10000] = 10000
        df[df < -10000] = -10000
        # 定义行业代码列表。
        industry_set = ['SWNLMY', 'SWMT', 'SWJCHG', 'SWGT', 'SWYSJS', 'SWDZ', 'SWJYDQ', 'SWSPYL', 'SWFZFS',
                        'SWQGZZ', 'SWYYSW', 'SWGYSY', 'SWJTYS', 'SWFDC', 'SWSMLS', 'SWSHFW', 'SWZH', 'SWJZCL',
                        'SWJZZS', 'SWDLSB', 'SWGFJG', 'SWXXFW1', 'SWCM', 'SWTX', 'SWYH', 'SWFYJR', 'SWQC', 'SWJXSB']
        for i in range(len(industry_set)):
            industry = get_code_list(industry_set[i],date=current_date).code.tolist()
            s = pd.Series([0] * len(df), index=df.index)
            s[list(set(industry) & set(s.index))] = 1
            df[industry_set[i]] = s

        # 选择特征和目标变量。
        X = df[['log_NC', 'LEV', 'NI_p', 'NI_n', 'g', 'log_RD', 'SWNLMY', 'SWMT', 'SWJCHG', 'SWGT', 'SWYSJS', 'SWDZ', 'SWJYDQ', 
                'SWSPYL', 'SWFZFS','SWQGZZ', 'SWYYSW', 'SWGYSY', 'SWJTYS', 'SWFDC', 'SWSMLS', 'SWSHFW', 'SWZH', 'SWJZCL',
                        'SWJZZS', 'SWDLSB', 'SWGFJG', 'SWXXFW1', 'SWCM', 'SWTX', 'SWYH', 'SWFYJR', 'SWQC', 'SWJXSB']]
        Y = df[['log_mcap']]
        # 填充特征和目标变量中的缺失值。
        X = X.fillna(0)
        Y = Y.fillna(0)
        # 创建支持向量回归模型。
        svr = SVR(kernel='rbf', gamma=0.1)
        # 训练模型。
        model = svr.fit(X, Y.values.ravel())
        # 计算因子值。
        factor = Y - pd.DataFrame(svr.predict(X), index=Y.index, columns=['log_mcap'])
        # 按因子值大小从小到大进行排序。
        factor = factor.sort_values(by='log_mcap')
        # 获取股票列表，并等权重分配资金
        stock_number = int(len(factor)/layer)
        # 获取前 stock_number 只股票作为目标股票
        targetlist = (factor.iloc[:stock_number]).copy()
        # 去除空值
        targetlist.dropna(inplace=True)  
        # 计算目标股票的持仓比例
        targetlist['target_percent'] = 1/len(targetlist)
        targetlist.index.name = 'code'
        targetlist['code'] = targetlist.index
        # 选择targetlist后15的行
        targetlist = targetlist.iloc[15:]
        context.targetlist = targetlist

        # 获取当前仓位列表
        mp = context.account().positions
        # 将股票代码转换为小写
        mp['code'] = mp['code'].apply(lambda x:str(x).lower()) 
        # 获取持仓股票列表
        mp_stock = mp[mp['volume_long'] > 0]
        # 调用股票委托操作函数
        handle_stock_orders(context,mp_stock)

####################################################################################################################
# 订单委托
def handle_stock_orders(context, mp_stock): # 股票调仓函数
    """
    根据当前持仓和目标持仓比例，生成调仓指令
    参数:
        context (object): 策略上下文对象
        mp_stock (DataFrame): 包含股票代码、当前持仓比例和目标持仓比例的DataFrame
    返回:
        None
    """
    # 获取当前仓位列表
    mp = context.account().positions
    # 将股票代码转换为小写
    mp['code'] = mp['code'].apply(lambda x:str(x).lower()) 
    # 遍历持仓股票列表，卖出不在选股列表中的股票。
    for stock in mp_stock['code'].tolist():
        if stock not in context.targetlist['code'].tolist():
            stock_sell = stock
            # 获取stock_sell的标的索引号
            stock_sell_index = mp[mp['code'] == stock_sell].index.tolist()[0]
            order_target_value(account_idx=0, target_idx=stock_sell_index, target_value=0, side=1, order_type=2,
                                price=0)  # 调整到目标仓位0，即平仓
            print(stock, '以市价卖出')
    # 计算需要买入的股票数量和每只股票的买入金额。
    if len(mp_stock['code'].tolist()) < layer:
        # 计算需要买入的股票数量
        stock_number = layer - len(mp_stock['code'].tolist())
        # 计算每只股票的买入金额
        csah = initial_cash / stock_number
    else:
        stock_number = 0
        csah = 0
    # 遍历选股列表，买入不在持仓股票列表中的股票。
    for stock in context.targetlist['code'].tolist():
        # 如果股票不在持仓股票列表中，且买入数量大于0，则买入。
        if stock not in mp_stock['code'].tolist() and stock_number > 0:
            stock_buy = stock
            # 获取stock_buy的标的索引号
            stock_buy_index = mp[mp['code'] == stock_buy].index.tolist()[0]
            # 买入
            order_value(account_idx=0, target_idx=stock_buy_index, value=csah, side=1, position_effect=1, order_type=2, price=0.0)
            print(stock, '以市价买入')
            stock_number -= 1
            if stock_number == 0:
                break 

####################################################################################################################
# 功能函数
def get_trade_date_list(begin_date, end_date, _period='monthly', begin_or_end='begin'): # 获取交易日列表
    # 获取开始日期到结束日期之间的所有交易日
    trade_date_list = get_trading_days('sse', begin_date, end_date)
    # 将交易日列表转换为Series对象
    time_series = pd.Series(trade_date_list)
    # 提取每个交易日的周数
    week = time_series.apply(lambda x: x.week)
    # 提取每个交易日的月份
    month = time_series.apply(lambda x: x.month)
    # 提取每个交易日的季度
    quarter = time_series.apply(lambda x: x.quarter)
    # 提取每个交易日的年份
    year = time_series.apply(lambda x: x.year)

    # 根据周期类型和开始或结束标志，生成交易日列表
    if _period == 'daily':
        # 如果周期类型为 'daily'，则直接返回所有交易日的日期字符串列表
        trade_date_list = time_series.apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    elif _period == 'weekly' and begin_or_end == 'begin':
        # 如果周期类型为 'weekly' 且开始或结束标志为 'begin'，则返回每周第一个交易日的日期字符串列表
        trade_date_list = time_series[week != week.shift(1)].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    elif _period == 'weekly' and begin_or_end == 'end':
        # 如果周期类型为 'weekly' 且开始或结束标志为 'end'，则返回每周最后一个交易日的日期字符串列表
        trade_date_list = time_series[week != week.shift(-1)].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    elif _period == 'monthly' and begin_or_end == 'begin':
        # 如果周期类型为 'monthly' 且开始或结束标志为 'begin'，则返回每月第一个交易日的日期字符串列表
        trade_date_list = time_series[month != month.shift(1)].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    elif _period == 'monthly' and begin_or_end == 'end':
        # 如果周期类型为 'monthly' 且开始或结束标志为 'end'，则返回每月最后一个交易日的日期字符串列表
        trade_date_list = time_series[month != month.shift(-1)].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    elif _period == 'quarterly' and begin_or_end == 'begin':
        # 如果周期类型为 'quarterly' 且开始或结束标志为 'begin'，则返回每季度第一个交易日的日期字符串列表
        trade_date_list = time_series[quarter != quarter.shift(1)].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    elif _period == 'quarterly' and begin_or_end == 'end':
        # 如果周期类型为 'quarterly' 且开始或结束标志为 'end'，则返回每季度最后一个交易日的日期字符串列表
        trade_date_list = time_series[quarter != quarter.shift(-1)].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    elif _period == 'yearly' and begin_or_end == 'begin':
        # 如果周期类型为 'yearly' 且开始或结束标志为 'begin'，则返回每年第一个交易日的日期字符串列表
        trade_date_list = time_series[year != year.shift(1)].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    elif _period == 'yearly' and begin_or_end == 'end':
        # 如果周期类型为 'yearly' 且开始或结束标志为 'end'，则返回每年最后一个交易日的日期字符串列表
        trade_date_list = time_series[year != year.shift(-1)].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()

    return trade_date_list

def get_recent_quarter_data_all(order_book_ids, date, field_bs=(), field_cf=(), field_ins=()): # 获取最近一个季度的数据
    """
    辅助函数，用于查询指定日期收盘后所能看到的最近一个季度的数据，同时获取资产负债表、现金流量表、利润表相关数据。

    参数:
    order_book_ids (str or list): 合约代码，可以是单个order_book_id或者order_book_id的列表，必填。
    date (str): 指定的查询日期，输入格式为“YYYY-MM-DD”，必填。
    field_bs (list): 需要返回的资产负债表相关财务字段，参考可选字段说明，为空则返回全部，可选。
    field_cf (list): 需要返回的现金流量表相关财务字段，参考可选字段说明，为空则返回全部，可选。
    field_ins (list): 需要返回的利润表相关财务字段，参考可选字段说明，为空则返回全部，可选。

    返回值:
    pandas.DataFrame: 包含查询结果（合并后的资产负债表、现金流量表、利润表数据）的数据框
    """
    # 将输入的日期字符串转换为datetime类型
    input_date = pd.to_datetime(date)

    # 获取资产负债表数据
    bs_data = fdmt.bs.bs_pit2(order_book_ids, start_quarter='2000q1', end_quarter='2099q4', publish_date='', field=field_bs)
    bs_filtered = bs_data[bs_data['publish_date'] <= input_date]
    if bs_filtered.empty:
        raise ValueError("没有找到资产负债表在指定日期之前发布的数据，请检查输入的日期或合约代码是否正确。")
    bs_filtered = bs_filtered.sort_values(by='quarter', ascending=False)
    bs_recent_quarter = bs_filtered.groupby('code').first().reset_index()

    # # 获取现金流量表数据
    # cf_data = at.fdmt.cf.cf_pit2(order_book_ids, start_quarter='2000q1', end_quarter='2099q4', publish_date='', field=field_cf)
    # cf_filtered = cf_data[cf_data['publish_date'] <= input_date]
    # if cf_filtered.empty:
    #     raise ValueError("没有找到现金流量表在指定日期之前发布的数据，请检查输入的日期或合约代码是否正确。")
    # cf_filtered.sort_values(by='quarter', ascending=False, inplace=True)
    # cf_recent_quarter = cf_filtered.groupby('code').first().reset_index()

    # 获取利润表数据
    ins_data = fdmt.ins.is_pit2(order_book_ids, start_quarter='2000q1', end_quarter='2099q4', publish_date='', field=field_ins)
    ins_filtered = ins_data[ins_data['publish_date'] <= input_date]
    if ins_filtered.empty:
        raise ValueError("没有找到利润表在指定日期之前发布的数据，请检查输入的日期或合约代码是否正确。")
    ins_filtered = ins_filtered.sort_values(by='quarter', ascending=False)
    ins_recent_quarter = ins_filtered.groupby('code').first().reset_index()

    # 合并三张表的数据
    # merged_data = pd.merge(bs_recent_quarter, cf_recent_quarter, on=['code', 'publish_date', 'quarter'], how='outer')
    # merged_data = pd.merge(merged_data, ins_recent_quarter, on=['code', 'publish_date', 'quarter'], how='outer')
    merged_data = pd.merge(bs_recent_quarter, ins_recent_quarter, on=['code', 'publish_date', 'quarter'], how='outer')
    return merged_data

def mad_winsorize_series(se, sigma_n): # 极值处理（绝对中位数差法）
    """
    对给定的序列进行极值处理（绝对中位数差法）

    参数:
        se (Series): 要处理的序列
        sigma_n (float): 标准差倍数，用于确定极值的阈值

    返回:
        Series: 处理后的序列
    """
    # 计算序列的中位数
    dm = se.median()
    # 计算序列与中位数的绝对差的中位数
    dm1 = (se - dm).abs().median()
    # 计算上界
    upper = dm + sigma_n * dm1
    # 计算下界
    lower = dm - sigma_n * dm1

    # 使用.loc 索引器来替换大于上界的值
    se.loc[se > upper] = upper
    # 使用.loc 索引器来替换小于下界的值
    se.loc[se < lower] = lower

    # 返回处理后的数据，保持原有的索引
    return pd.Series(index=se.index, data=se.ravel())

def winsorize_series(se, limits=0.025, inclusive=False): # 极值处理（缩尾处理）
    """
    对给定的序列进行极值处理（缩尾处理）

    参数:
        se (Series): 要处理的序列
        limits (float, optional): 极值处理的阈值，默认为0.025
        inclusive (bool, optional): 是否包含阈值，默认为False

    返回:
        Series: 处理后的序列
    """
    # 设置极值处理的上下限
    limits = [limits, limits]
    # 设置是否包含阈值
    inclusive = [inclusive, inclusive]
    # 使用 mstats.winsorize 函数进行极值处理
    data = mstats.winsorize(se, limits, inclusive)
    # 返回处理后的数据，保持原有的索引
    return pd.Series(index=se.index, data=data.ravel())

def standardize_series(se): # 标准化处理
    """
    对给定的序列进行标准化处理

    参数:
        se (Series): 要处理的序列

    返回:
        Series: 处理后的序列
    """
    # 计算序列的标准差
    se_std = se.std()
    # 计算序列的均值
    se_mean = se.mean()
    # 标准化处理：(序列 - 均值) / 标准差
    return (se - se_mean) / se_std

####################################################################################################################
# 整体回测参数
universe= 'HS300'    # 选择投资域
begin_date = '2023-09-01'    # 回测开始时间
end_date = '2024-12-08'      # 回测结束时间
initial_cash = 10000000     # 回测金额
benchmark = 'HS300'          # 选择基准

# 多因子参数设置
freq = ['weekly','begin']    # 股票刷新频率（daily为日，weekly为周，monthly为月，quarterly为季，yearly为年，begin为第一个交易日，end为最后一个交易日）
factor_list = ['market_cap_2','inc_revenue_lyr']    # 选择因子
factor_processing = {0:{'winsorize':[True,1], 'standardize':True},
                     1:{'winsorize':[True,1], 'standardize':True},
                   }# 各个因子处理
X_list = ['log_NC', 'NI_p', 'NI_n', 'log_RD']
X_processing = {'log_NC':{'winsorize':[True,1], 'standardize':True},
                'NI_p':{'winsorize':[True,1], 'standardize':True},
                'NI_n':{'winsorize':[True,1], 'standardize':True},
                'log_RD':{'winsorize':[True,1], 'standardize':True},
                }# 各个财务数据处理
direction = [1,1]    # 因子方向，-1为负向，1为正向
layer = 10           # 分成几层


if __name__ == '__main__':
    target = get_code_list_set(universe,begin_date,end_date)['code'].tolist()
    strategy_name = '策略:'+str(factor_list) + '+' + universe
    # 策略回测，strategy_name---策略名，file_path---策略文件路径，target_list---标的资产列表，frequency---策略刷新频率
    # fre_num---策略刷新频数，begin_date---回测开始日期，end_data---回测结束日期，fq---复权类型（1为前复权，2为后复权）
    run_backtest(strategy_name=strategy_name,file_path='.',target_list=target,frequency='day',fre_num=1,begin_date=begin_date,end_date=end_date,fq=1)