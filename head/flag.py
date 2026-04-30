# globals_task.py
TASK_STATE = 1  # 0 = 背景, 1 = 道路
ROAD_COUNT = 1  # 道路计数器，用于跟踪连续道路次数
TEST_TASK_STATE = 1  # 0 = 背景, 1 = 道路

def switch_task_state():
    """切换全局任务状态，实现2次道路->1次背景的循环"""
    global TASK_STATE, ROAD_COUNT
    if TASK_STATE == 0:  # 当前是背景
        TASK_STATE = 1  # 切换到道路
        ROAD_COUNT = 1  # 道路计数设为1
    elif TASK_STATE == 1:  # 当前是道路
        if ROAD_COUNT < 2:  # 如果道路次数小于2
            ROAD_COUNT += 1  # 继续道路，计数+1
        else:  # 如果已经2次道路了
            TASK_STATE = 0  # 切换到背景
            ROAD_COUNT = 0  # 重置道路计数

def get_task_state():
    """获取当前任务状态"""
    return TASK_STATE

def get_test_task_state():
    """获取当前测试任务状态"""
    return TEST_TASK_STATE

def switch_test_task_state():
    """切换全局测试任务状态（0 <-> 1）"""
    global TEST_TASK_STATE
    TEST_TASK_STATE = 1 - TEST_TASK_STATE
