"""
精简高级瞄准系统
保留核心功能：状态机锁定、人类行为模拟、智能目标选择
"""
import time
import math
import random
from collections import deque
from typing import Optional, Tuple, List, Dict
from enum import Enum
from dataclasses import dataclass

class LockState(Enum):
    SEARCHING = "searching"
    ACQUIRING = "acquiring"
    LOCKED = "locked"
    TRACKING = "tracking"
    LOST = "lost"

@dataclass
class TargetInfo:
    x: float
    y: float
    width: float
    height: float
    confidence: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    last_seen: float = 0.0

class AdvancedAimingSystem:
    def __init__(self):
        self.lock_state = LockState.SEARCHING
        self.current_target: Optional[TargetInfo] = None
        self.state_start_time = time.time()
        self.lock_start_time = 0.0

        # 历史记录
        self.target_history = deque(maxlen=10)
        self.move_history = deque(maxlen=5)

        # 平滑状态
        self.smoothed_x = 0.0
        self.smoothed_y = 0.0

        # 性能统计
        self.stats = {
            'targets_locked': 0,
            'target_switches': 0,
            'reaction_times': []
        }

    def update(self, targets: List[Dict], screen_center: Tuple[int, int],
               config) -> Optional[Tuple[float, float]]:
        """主更新函数"""
        current_time = time.time()

        # 状态机处理
        old_state = self.lock_state
        movement = None
        if self.lock_state == LockState.SEARCHING:
            movement = self._handle_searching(targets, screen_center, config, current_time)
        elif self.lock_state == LockState.ACQUIRING:
            movement = self._handle_acquiring(targets, screen_center, config, current_time)
        elif self.lock_state == LockState.LOCKED:
            movement = self._handle_locked(targets, screen_center, config, current_time)
        elif self.lock_state == LockState.TRACKING:
            movement = self._handle_tracking(targets, screen_center, config, current_time)
        elif self.lock_state == LockState.LOST:
            movement = self._handle_lost(current_time)

        # 调试信息
        if getattr(config, 'cpsDisplay', False):
            if old_state != self.lock_state:
                print(f"状态转换: {old_state.value} -> {self.lock_state.value}")
            if movement:
                print(f"状态 {self.lock_state.value}: 移动 {movement}")

        return movement

    def _handle_searching(self, targets: List[Dict], screen_center: Tuple[int, int],
                         config, current_time: float) -> Optional[Tuple[float, float]]:
        """搜索状态"""
        if not targets:
            return None

        # 选择最佳目标（距离优先）
        best_target = min(targets, key=lambda t: math.sqrt(
            (t['current_mid_x'] - screen_center[0])**2 +
            (t['current_mid_y'] - screen_center[1])**2
        ))

        # 开始获取目标
        self.current_target = TargetInfo(
            x=best_target['current_mid_x'],
            y=best_target['current_mid_y'],
            width=best_target['width'],
            height=best_target['height'],
            confidence=best_target['confidence'],
            velocity_x=0.0,  # 显式初始化速度
            velocity_y=0.0,
            last_seen=current_time
        )

        self.lock_state = LockState.ACQUIRING
        self.state_start_time = current_time

        # 模拟人类反应延迟
        reaction_delay = getattr(config, 'humanReactionTime', 0.15)
        self.lock_acquisition_time = current_time + reaction_delay

        return None

    def _handle_acquiring(self, targets: List[Dict], screen_center: Tuple[int, int],
                         config, current_time: float) -> Optional[Tuple[float, float]]:
        """获取锁定状态"""
        # 反应延迟检查
        if current_time < self.lock_acquisition_time:
            return None

        # 查找当前目标
        current_target = self._find_target(targets, max_distance=50)
        if not current_target:
            self._transition_to_lost()
            return None

        # 更新目标信息
        self._update_target(current_target, current_time)

        # 计算距离
        distance = math.sqrt(
            (self.current_target.x - screen_center[0])**2 +
            (self.current_target.y - screen_center[1])**2
        )

        # 距离足够近则锁定 (增大锁定距离阈值)
        if distance < 50:  # 从20增加到50像素
            self.lock_state = LockState.LOCKED
            self.lock_start_time = current_time
            self.stats['targets_locked'] += 1

            # 记录反应时间
            reaction_time = current_time - self.state_start_time
            self.stats['reaction_times'].append(reaction_time)

        # 计算瞄准移动
        return self._calculate_movement(screen_center, config, acquiring=True)

    def _handle_locked(self, targets: List[Dict], screen_center: Tuple[int, int],
                      config, current_time: float) -> Optional[Tuple[float, float]]:
        """锁定状态"""
        current_target = self._find_target(targets, max_distance=30)
        if not current_target:
            self._transition_to_lost()
            return None

        self._update_target(current_target, current_time)

        # 检查目标是否移动
        speed = math.sqrt(self.current_target.velocity_x**2 + self.current_target.velocity_y**2)
        if speed > 25:
            self.lock_state = LockState.TRACKING

        # 反检测：最大锁定时间
        max_lock_time = getattr(config, 'maxContinuousFrames', 120) / 60.0  # 转换为秒
        if current_time - self.lock_start_time > max_lock_time:
            self._transition_to_lost()
            return None

        return self._calculate_movement(screen_center, config)

    def _handle_tracking(self, targets: List[Dict], screen_center: Tuple[int, int],
                        config, current_time: float) -> Optional[Tuple[float, float]]:
        """跟踪状态"""
        current_target = self._find_target(targets, max_distance=40)
        if not current_target:
            self._transition_to_lost()
            return None

        self._update_target(current_target, current_time)

        # 检查目标是否停止
        speed = math.sqrt(self.current_target.velocity_x**2 + self.current_target.velocity_y**2)
        if speed < 15:
            self.lock_state = LockState.LOCKED
            self.lock_start_time = current_time

        return self._calculate_movement(screen_center, config, tracking=True)

    def _handle_lost(self, current_time: float) -> Optional[Tuple[float, float]]:
        """丢失状态"""
        self.current_target = None

        # 短暂暂停后重新搜索
        if current_time - self.state_start_time > 0.3:
            self.lock_state = LockState.SEARCHING
            self.state_start_time = current_time

        return None

    def _calculate_movement(self, screen_center: Tuple[int, int], config,
                           acquiring: bool = False, tracking: bool = False) -> Tuple[float, float]:
        """计算瞄准移动"""
        if not self.current_target:
            return 0.0, 0.0

        # 目标位置
        target_x = self.current_target.x
        target_y = self.current_target.y

        # 应用瞄准偏移
        aim_offset = getattr(config, 'aimOffsetRatio', 0.4)
        y_offset = (aim_offset - 0.5) * self.current_target.height
        target_y += y_offset

        # 运动预测
        if tracking and getattr(config, 'enablePredictiveAim', True):
            pred_strength = getattr(config, 'predictionStrength', 0.3)
            target_x += self.current_target.velocity_x * pred_strength * 0.1
            target_y += self.current_target.velocity_y * pred_strength * 0.1

        # 原始移动量
        raw_x = target_x - screen_center[0]
        raw_y = target_y - screen_center[1]

        # 人类化微调
        if random.random() < 0.3:  # 30%概率添加微调
            raw_x += random.uniform(-0.5, 0.5)
            raw_y += random.uniform(-0.5, 0.5)

        # 平滑处理
        smooth_factor = getattr(config, 'smoothingFactor', 0.6)
        if acquiring:
            smooth_factor *= 0.7  # 获取时减少平滑
        elif tracking:
            smooth_factor *= 0.8  # 跟踪时中等平滑

        self.smoothed_x = self.smoothed_x * smooth_factor + raw_x * (1 - smooth_factor)
        self.smoothed_y = self.smoothed_y * smooth_factor + raw_y * (1 - smooth_factor)

        # 移动阈值 (降低阈值，让小距离移动也能生效)
        threshold = getattr(config, 'movementThreshold', 0.5)  # 从1.0降到0.5
        distance = math.sqrt(self.smoothed_x**2 + self.smoothed_y**2)

        if distance < threshold:
            return 0.0, 0.0

        return self.smoothed_x, self.smoothed_y

    def _find_target(self, targets: List[Dict], max_distance: float) -> Optional[Dict]:
        """在列表中查找当前目标"""
        if not self.current_target or not targets:
            return None

        for target in targets:
            distance = math.sqrt(
                (target['current_mid_x'] - self.current_target.x)**2 +
                (target['current_mid_y'] - self.current_target.y)**2
            )
            if distance <= max_distance:
                return target

        return None

    def _update_target(self, target_dict: Dict, current_time: float):
        """更新目标信息"""
        if not self.current_target:
            return

        # 保存旧位置用于速度计算
        old_x, old_y = self.current_target.x, self.current_target.y
        old_time = self.current_target.last_seen

        # 先更新位置
        self.current_target.x = target_dict['current_mid_x']
        self.current_target.y = target_dict['current_mid_y']
        self.current_target.width = target_dict['width']
        self.current_target.height = target_dict['height']
        self.current_target.confidence = target_dict['confidence']
        self.current_target.last_seen = current_time

        # 计算速度（使用新旧位置差）
        dt = current_time - old_time
        if dt > 0 and dt < 1.0:  # 防止异常大的时间间隔
            self.current_target.velocity_x = (self.current_target.x - old_x) / dt
            self.current_target.velocity_y = (self.current_target.y - old_y) / dt
        else:
            # 时间间隔异常，重置速度
            self.current_target.velocity_x = 0.0
            self.current_target.velocity_y = 0.0

    def _transition_to_lost(self):
        """转换到丢失状态"""
        self.lock_state = LockState.LOST
        self.state_start_time = time.time()
        self.stats['target_switches'] += 1

        # 重置平滑状态
        self.smoothed_x = 0.0
        self.smoothed_y = 0.0

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'current_state': self.lock_state.value,
            'targets_locked': self.stats['targets_locked'],
            'target_switches': self.stats['target_switches'],
            'avg_reaction_time': sum(self.stats['reaction_times']) / len(self.stats['reaction_times']) if self.stats['reaction_times'] else 0
        }

# 全局实例
aiming_system = AdvancedAimingSystem()

def get_advanced_aiming_movement(targets_df, screen_center_x: int, screen_center_y: int, config) -> Optional[Tuple[float, float]]:
    """获取高级瞄准移动量"""
    # 转换数据格式
    targets = []
    if not targets_df.empty:
        for _, row in targets_df.iterrows():
            targets.append({
                'current_mid_x': row['current_mid_x'],
                'current_mid_y': row['current_mid_y'],
                'width': row['width'],
                'height': row['height'],
                'confidence': row['confidence']
            })

    screen_center = (screen_center_x, screen_center_y)

    # 调试信息
    if getattr(config, 'cpsDisplay', False):
        print(f"瞄准系统输入: 目标数={len(targets)}, 屏幕中心={screen_center}")
        if targets:
            closest_target = min(targets, key=lambda t: math.sqrt((t['current_mid_x'] - screen_center[0])**2 + (t['current_mid_y'] - screen_center[1])**2))
            distance = math.sqrt((closest_target['current_mid_x'] - screen_center[0])**2 + (closest_target['current_mid_y'] - screen_center[1])**2)
            print(f"最近目标: 位置=({closest_target['current_mid_x']:.1f}, {closest_target['current_mid_y']:.1f}), 距离={distance:.1f}")

    # 更新瞄准系统
    result = aiming_system.update(targets, screen_center, config)

    if getattr(config, 'cpsDisplay', False) and result:
        print(f"瞄准系统输出: 移动量=({result[0]:.2f}, {result[1]:.2f})")

    return result