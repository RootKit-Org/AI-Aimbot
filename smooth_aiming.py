"""
优化的瞄准逻辑模块
包含平滑移动、目标锁定、抖动抑制等功能
"""
import numpy as np
import time
from collections import deque
import math

class SmoothAiming:
    def __init__(self):
        """初始化平滑瞄准系统"""
        self.last_target = None
        self.locked_target_id = None
        self.lock_frame_count = 0
        self.movement_history = deque(maxlen=5)  # 保存最近5次移动
        self.target_history = deque(maxlen=5)    # 增加历史记录数量以提高预测精度
        self.last_move_time = time.time()
        self.smooth_x = 0.0
        self.smooth_y = 0.0
        self.frame_times = deque(maxlen=10)      # 监控帧率性能
        self.input_lag_compensation = 0.016      # 输入延迟补偿（约16ms）
        
    def calculate_smooth_movement(self, target_x, target_y, center_x, center_y, config):
        """
        计算平滑的鼠标移动，使用改进的算法
        
        Args:
            target_x, target_y: 目标坐标
            center_x, center_y: 屏幕中心坐标
            config: 配置对象
            
        Returns:
            tuple: (smooth_x, smooth_y) 平滑后的移动坐标
        """
        # 计算原始移动量
        raw_x = target_x - center_x
        raw_y = target_y - center_y
        
        # 如果启用平滑移动
        if getattr(config, 'enableSmoothMovement', True):
            smoothing_factor = getattr(config, 'smoothingFactor', 0.25)
            
            # 计算移动距离，用于自适应平滑
            move_distance = math.sqrt(raw_x * raw_x + raw_y * raw_y)
            
            # 自适应平滑：距离越大，平滑程度越小（响应更快）
            if move_distance > 50:  # 大距离移动
                adaptive_factor = smoothing_factor * 0.5  # 减少平滑，快速跟进
            elif move_distance > 20:  # 中等距离
                adaptive_factor = smoothing_factor * 0.7
            else:  # 小距离移动
                adaptive_factor = smoothing_factor  # 正常平滑
            
            # 使用自适应指数移动平均进行平滑
            self.smooth_x = self.smooth_x * adaptive_factor + raw_x * (1 - adaptive_factor)
            self.smooth_y = self.smooth_y * adaptive_factor + raw_y * (1 - adaptive_factor)
            
            return self.smooth_x, self.smooth_y
        
        return raw_x, raw_y
    
    def apply_movement_threshold(self, move_x, move_y, config):
        """
        应用移动阈值，忽略微小移动
        
        Args:
            move_x, move_y: 移动量
            config: 配置对象
            
        Returns:
            tuple: 过滤后的移动量
        """
        threshold = getattr(config, 'movementThreshold', 2)
        
        # 计算移动距离
        move_distance = math.sqrt(move_x * move_x + move_y * move_y)
        
        # 如果移动距离小于阈值，则不移动
        if move_distance < threshold:
            return 0, 0
            
        return move_x, move_y
    
    def select_target_with_lock(self, targets, config):
        """
        带锁定功能的目标选择
        
        Args:
            targets: 目标数据框
            config: 配置对象
            
        Returns:
            dict: 选中的目标信息，如果没有目标则返回None
        """
        if len(targets) == 0:
            self._reset_lock()
            return None
        
        lock_frames = getattr(config, 'targetLockFrames', 15)
        max_lock_distance = getattr(config, 'maxLockDistance', 100)
        
        # 如果有锁定的目标，尝试保持锁定
        if self.locked_target_id is not None and self.lock_frame_count < lock_frames:
            locked_target = self._find_locked_target(targets, max_lock_distance)
            if locked_target is not None:
                self.lock_frame_count += 1
                return locked_target
            else:
                self._reset_lock()
        
        # 选择新目标
        selected_target = self._select_best_target(targets, config)
        if selected_target is not None:
            self._lock_new_target(selected_target)
        
        return selected_target
    
    def _find_locked_target(self, targets, max_distance):
        """查找已锁定的目标"""
        if self.last_target is None:
            return None
        
        last_x, last_y = self.last_target['x'], self.last_target['y']
        
        for i, target in targets.iterrows():
            current_x = target['current_mid_x']
            current_y = target['current_mid_y']
            
            # 计算与上次目标的距离
            distance = math.sqrt((current_x - last_x)**2 + (current_y - last_y)**2)
            
            if distance <= max_distance:
                return {
                    'x': current_x,
                    'y': current_y,
                    'width': target['width'],
                    'height': target['height'],
                    'confidence': target['confidence']
                }
        
        return None
    
    def _select_best_target(self, targets, config):
        """选择最佳目标"""
        # 如果启用中心优先，按距离中心的距离排序
        if getattr(config, 'centerOfScreen', True):
            best_target = targets.iloc[0]  # 已经按距离中心排序
        else:
            # 按置信度选择
            best_target = targets.loc[targets['confidence'].idxmax()]
        
        return {
            'x': best_target['current_mid_x'],
            'y': best_target['current_mid_y'], 
            'width': best_target['width'],
            'height': best_target['height'],
            'confidence': best_target['confidence']
        }
    
    def _lock_new_target(self, target):
        """锁定新目标"""
        self.locked_target_id = id(target)
        self.lock_frame_count = 0
        self.last_target = target.copy()
    
    def _reset_lock(self):
        """重置锁定状态"""
        self.locked_target_id = None
        self.lock_frame_count = 0
        self.last_target = None
    
    def apply_predictive_aiming(self, target, config):
        """
        应用预测性瞄准，使用改进的运动预测算法
        
        Args:
            target: 目标信息
            config: 配置对象
            
        Returns:
            tuple: 预测后的目标坐标
        """
        if not getattr(config, 'enablePredictiveAim', True):
            return target['x'], target['y']
        
        current_time = time.time()
        current_pos = (target['x'], target['y'], current_time)
        
        # 记录目标历史位置
        self.target_history.append(current_pos)
        
        # 需要至少2个历史点来计算速度
        if len(self.target_history) < 2:
            return target['x'], target['y']
        
        # 计算多点平均速度以提高预测精度
        velocities_x = []
        velocities_y = []
        
        for i in range(1, len(self.target_history)):
            prev_pos = self.target_history[i-1]
            curr_pos = self.target_history[i]
            
            dt = curr_pos[2] - prev_pos[2]
            if dt > 0:
                vx = (curr_pos[0] - prev_pos[0]) / dt
                vy = (curr_pos[1] - prev_pos[1]) / dt
                velocities_x.append(vx)
                velocities_y.append(vy)
        
        if not velocities_x:
            return target['x'], target['y']
        
        # 使用加权平均（最近的速度权重更大）
        weights = [i + 1 for i in range(len(velocities_x))]
        avg_vx = sum(vx * w for vx, w in zip(velocities_x, weights)) / sum(weights)
        avg_vy = sum(vy * w for vy, w in zip(velocities_y, weights)) / sum(weights)
        
        # 计算预测时间（根据速度自适应调整）
        speed = math.sqrt(avg_vx * avg_vx + avg_vy * avg_vy)
        prediction_strength = getattr(config, 'predictionStrength', 0.6)
        
        # 根据移动速度调整预测时间
        if speed > 200:  # 快速移动
            prediction_time = prediction_strength * 0.08
        elif speed > 100:  # 中速移动
            prediction_time = prediction_strength * 0.06
        else:  # 慢速或静止
            prediction_time = prediction_strength * 0.04
        
        # 预测未来位置
        predicted_x = current_pos[0] + avg_vx * prediction_time
        predicted_y = current_pos[1] + avg_vy * prediction_time
        
        return predicted_x, predicted_y
    
    def calculate_aimpoint_offset(self, target, config):
        """
        计算瞄准点偏移，基于用户配置的精确位置
        
        Args:
            target: 目标信息
            config: 配置对象
            
        Returns:
            tuple: (x_offset, y_offset) 瞄准点偏移量
        """
        box_height = target['height']
        box_width = target['width']
        
        # 获取用户配置的瞄准偏移比例（默认0.4，即从顶部向下40%）
        aim_offset_ratio = getattr(config, 'aimOffsetRatio', 0.4)
        
        # X轴偏移：始终瞄准中轴线（水平中心），所以X偏移为0
        x_offset = 0.0
        
        # Y轴偏移：从目标框顶部向下的偏移量
        # 注意：在屏幕坐标系中，Y轴向下为正，所以我们需要从中心点向上偏移
        # target['y'] 是目标框的中心点，我们需要计算到指定比例位置的偏移
        y_offset_from_center = (aim_offset_ratio - 0.5) * box_height
        
        # 如果是头部模式，可以进一步微调
        if getattr(config, 'headshot_mode', False):
            # 头部模式下，稍微向上调整瞄准点
            y_offset_from_center -= box_height * 0.05
        
        return x_offset, y_offset_from_center

# 全局平滑瞄准实例
smooth_aiming = SmoothAiming()

def get_optimized_mouse_movement(targets, center_x, center_y, config):
    """
    获取优化的鼠标移动量，使用精确瞄准逻辑
    
    Args:
        targets: 检测到的目标数据框
        center_x, center_y: 屏幕中心坐标
        config: 配置对象
        
    Returns:
        tuple: (move_x, move_y) 或 None 如果没有有效移动
    """
    global smooth_aiming
    
    # 选择目标（带锁定功能）
    selected_target = smooth_aiming.select_target_with_lock(targets, config)
    
    if selected_target is None:
        return None
    
    # 应用预测性瞄准得到基础目标位置
    predicted_x, predicted_y = smooth_aiming.apply_predictive_aiming(selected_target, config)
    
    # 计算精确瞄准点偏移（中轴线上方0.4位置）
    x_offset, y_offset = smooth_aiming.calculate_aimpoint_offset(selected_target, config)
    
    # 应用偏移到预测位置
    # X轴：使用预测的X坐标 + X偏移（通常为0，保持中轴线）
    # Y轴：使用预测的Y坐标 + Y偏移（向上偏移到0.4位置）
    target_x = predicted_x + x_offset
    target_y = predicted_y + y_offset
    
    # 计算平滑移动
    smooth_x, smooth_y = smooth_aiming.calculate_smooth_movement(
        target_x, target_y, center_x, center_y, config)
    
    # 应用移动阈值
    final_x, final_y = smooth_aiming.apply_movement_threshold(smooth_x, smooth_y, config)
    
    return final_x, final_y