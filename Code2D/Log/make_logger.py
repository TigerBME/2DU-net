from .logger import TrainingLogger

def make_logger(log_config: dict):
    '''
    {
        'log_path': 'path/to/log/file',
        'metrics_to_log':
        {
            'train_loss':true
            'train_accuracy':false
            'test_loss':true
            'test_accuracy':false
        }  
    }
    '''
    metrics = log_config.pop('metrics_to_log')
    metrics_to_log = []
    for metric in metrics:
        if metrics[metric]:
            metrics_to_log.append(metric)
    log_config['csv_columns'] = metrics_to_log

    logger = TrainingLogger(log_config)
    return logger
            
    
        

