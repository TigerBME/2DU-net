import torch

class EarlyStopper:
    """
    早停机制专用类
    """
    def __init__(self, es_cfg: dict):
        """
        config格式：
        {
            "early_stopping": {
                "patience": 10,     # 容忍不进步的epoch数
                "delta": 0.001,     # 视为有效改进的最小变化量
                "mode": "max",      # 优化方向：max（越大越好）/min（越小越好）
                "model_save_path": "best_model.pth"  # 模型保存路径
            }
        }
        """
        must_keys = ['patience', 'delta','mode','model_save_path']
        for key in must_keys:
            if key not in es_cfg:
                raise ValueError(f"Early stopping config must have key: {key}")
        self.patience = es_cfg['patience']
        self.delta = es_cfg['delta']
        self.mode = es_cfg['mode']
        self.save_path = es_cfg['model_save_path']
        
        # 初始化状态变量
        self.best_score = None
        self.best_epoch = 0
        self.epoch = 0
        self.counter = 0
        self.early_stop_triggered = False
        
        # 设置比较逻辑
        if self.mode == 'max':
            self._comparator = lambda current, best: current > (best + self.delta)
            self._progress = lambda current, best: current - best
        elif self.mode == 'min':
            self._comparator = lambda current, best: current < (best - self.delta)
            self._progress = lambda current, best: best - current
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'max' or 'min'")

    def check_early_stop(self, monitor_metrics: float, model: torch.nn.Module) -> dict:
        """
        检查当前epoch是否需要早停
        返回：bool类型，True表示触发早停
        """
        self.epoch += 1        

        if self.best_score is None: # epoch 1
            self._update_best(monitor_metrics, model)
            self.progress = 0.0
        else:
            self.progress = self.get_progress(monitor_metrics)

            if self._comparator(monitor_metrics, self.best_score):
                self._update_best(monitor_metrics, model)
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop_triggered = True

        return_config= {
            "early_stop": self.early_stop_triggered,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "patience_left": self.patience - self.counter,
            "progress": self.progress
        }
        
        return return_config

    def _update_best(self, score: float, model: torch.nn.Module):
        """更新最佳分数并保存模型"""
        self.best_score = score
        self.best_epoch = self.epoch
        torch.save(model.state_dict(), self.save_path)
        print(f"Best score updated: {self.best_score} at epoch {self.best_epoch}")

    def get_progress(self, current_score: float) -> float:
        """获取当前相对于最佳值的改进量"""
        if self.best_score is None:
            return float('inf') if self.mode == 'max' else float('-inf')
        return self._progress(current_score, self.best_score)
