# globals_task.py
TASK_STATE = 1  # 0 = 背景, 1 = 前景
TEST_TASK_STATE = 1  # 0 = 背景, 1 = 前景

def switch_task_state():
    """切换全局任务状态（0 <-> 1）"""
    global TASK_STATE
    TASK_STATE = 1 - TASK_STATE

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
