import os
import csv
from datetime import datetime
import shutil
import matplotlib.pyplot as plt

class TrainingLogger:
    """
    训练记录专用类，负责日志和指标记录
    """
    def __init__(self, log_cfg: dict):
        """
        根据配置初始化日志系统
        """
        self.log_dir = log_cfg['log_dir']
        self.overwrite = log_cfg.get('overwrite', False)
        
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化文件路径
        self.txt_path = os.path.join(self.log_dir, "training_log.txt")
        self.csv_path = os.path.join(self.log_dir, "training_metrics.csv")
        
        self.txt_init = False
        self.csv_init = False

        # 初始化日志文件
        self._init_txt()

    # def _init_files(self):
    #     """初始化日志文件结构"""
    #     # 文本日志
    #     if not os.path.exists(self.txt_path) or self.overwrite:
    #         with open(self.txt_path, 'w') as f:
    #             f.write("Training Log\n")
    #             f.write("="*40 + "\n")
        
    #     # CSV指标记录
    #     if not os.path.exists(self.csv_path) or self.overwrite:
    #         with open(self.csv_path, 'w', newline='') as f:
    #             writer = csv.writer(f)
    #             writer.writerow(self.csv_columns)

    def _init_txt(self):
        """初始化日志文件结构"""
        # 文本日志
        if self.txt_init == True:
            # 已经初始化过了，不用再初始化
            return
        
        if not os.path.exists(self.txt_path) or self.overwrite:
            with open(self.txt_path, 'w') as f:
                f.write("Training Log\n")
                f.write("="*40 + "\n")
            self.txt_init = True

    def _init_csv(self, metrics: dict):
        '''
        metrics: 训练过程中需要记录的指标
        '''
        if self.csv_init == True:
            # 已经初始化过了，不用再初始化
            return
        
        """初始化训练过程记录文件结构"""
        # CSV指标记录
        self.csv_columns = [metric for metric in metrics]
        if not os.path.exists(self.csv_path) or self.overwrite:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_columns)

            self.csv_init = True

    def log_metrics(self, metrics: dict):
        """记录指标到CSV文件"""
        if not self.csv_init:
            # 如果没有初始化过csv文件，先初始化
            self._init_csv(metrics)

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_columns)
            
            # 过滤 metrics，只保留那些在 self.csv_columns 中的键
            filtered_metrics = {key: f"{value:.6f}" if isinstance(value, float) else value
                                for key, value in metrics.items() }
           
            writer.writerow(filtered_metrics)

    def log_message(self, message: str):
        """记录消息到文本日志"""
        with open(self.txt_path, 'a') as f:
            f.write(f"[{datetime.now()}]: {message}\n")

    def start_experiment(self, cfg_path: str):
        """记录实验开始信息，并将配置文件复制到日志文件夹内"""
        # 复制配置文件到日志目录
        cfg_name = os.path.basename(cfg_path)
        aim_cfg_path = os.path.join(self.log_dir, cfg_name)
        shutil.copy(cfg_path, aim_cfg_path)
        
        self.log_message("Experiment Started")
        self.log_message(f"Configuration file copied to: {aim_cfg_path}")
        self.log_message("-"*40)

    def end_experiment(self, summary: str):
        """记录实验结束信息，并绘制训练曲线"""
        self.log_message("-"*40)
        self.log_message(f"Experiment Completed: {summary}")
        self._draw_train_log()

    def _draw_train_log(self):
        """绘制训练曲线"""
        # 读取csv文件中的数据
        data = {}
        with open(self.csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                for column in self.csv_columns:
                    if column not in data:
                        data[column] = []
                    if column in row:
                        data[column].append(float(row[column]) if column != 'epoch' else int(row[column]))
        
        if not os.path.exists(os.path.join(self.log_dir, 'process_curve')):
            os.makedirs(os.path.join(self.log_dir, 'process_curve'))

        # 遍历可以绘制的列
        for column in data:
            if column != 'epoch':
                # 创建一个新的图形
                plt.figure()
                plt.plot(data['epoch'], data[column], label=column)
                plt.xlabel('Epoch')
                plt.ylabel(column.capitalize())
                plt.title(f'Training {column.capitalize()} per Epoch')
                plt.legend()
                # plt.xticks(ticks=data['epoch'].astype(int), labels=data['epoch'].astype(int))
                # 保存图形到文件
                plt.savefig(os.path.join(self.log_dir, f'process_curve/{column}.png'))
                plt.close()


