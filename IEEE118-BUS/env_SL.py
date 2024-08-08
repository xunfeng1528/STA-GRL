import random
from datetime import datetime, timedelta
import gen_Data
import time


def gen_RL_GNN_data(GTrans_data, split, day_num, day):
    num = int(288/split)
    Gdata = GTrans_data[day_num*num:(day_num+1)*num]
    return Gdata


def get_day_num(start_num: int):
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 12, 31)
    delta = end_date - start_date
    Gdata_list_train, Gdata_list_test, split, test_indices = gen_Data.gen_GNN_data()
    # 生成日期列表
    all_dates = [start_date + timedelta(days=i) for i in range(delta.days + 1)]
    # 移除索引为 test_indices 的日期
    remaining_dates = [date for i, date in enumerate(all_dates) if i not in test_indices]
    # 重新编排索引
    remaining_dates_indexed = {i: date for i, date in enumerate(remaining_dates)}
    # 随机选择一个日期
    random.seed(time.time())
    random_index = random.randint(0, len(remaining_dates) - 3)
    random.seed(50)
    random_day = remaining_dates_indexed[random_index]
    day_of_year = random_day.timetuple().tm_yday
    day_num = day_of_year - start_num
    day = random_day.strftime("%Y/%m/%d")
    Gdata = gen_RL_GNN_data(Gdata_list_train, split, day_num, day)
    return day_num, day, Gdata, split


def gen_initstate(start_num: int, n_units, device):
    day_num, day, Gdata, split = get_day_num(start_num)
    period_demand = [item.x.sum(dim=0).cpu().detach().numpy() for item in Gdata]
    return period_demand, day_num, day, Gdata

