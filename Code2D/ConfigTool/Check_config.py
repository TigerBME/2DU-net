from datetime import datetime

def check_config(config):
    train_data_size = len(config['data']['train']['data']) * config['data']['train']['data_per_picture']
    val_data_size = len(config['data']['val']['data']) * config['data']['val']['data_per_picture']
    test_data_size = len(config['data']['test']['data']) * config['data']['test']['data_per_picture']
    train_dataloader_size = train_data_size // config['data']['train']['dataloader_args']['batch_size']
    val_dataloader_size = val_data_size // config['data']['val']['dataloader_args']['batch_size']
    test_dataloader_size = test_data_size // config['data']['test']['dataloader_args']['batch_size']

    config['statistics'] = {
        'train_data_size' : train_data_size,
        'val_data_size' : val_data_size,
        'test_data_size' : test_data_size,
        'train_dataloader_size' : train_dataloader_size,
        'val_dataloader_size' : val_dataloader_size,
        'test_dataloader_size' : test_dataloader_size,
        'Creat_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # 数据集检查
    if len(config['data']['train']['data']) == 0:
        print("训练集为空！")
    else:
        print(f"训练集大小：{len(config['data']['train']['data'])} * \
{config['data']['train']['data_per_picture']} = \
{len(config['data']['train']['data'] * config['data']['train']['data_per_picture'])}")
    if len(config['data']['val']['data']) == 0:
        print("验证集为空！")
    else:
        print(f"验证集大小：{len(config['data']['val']['data'])}")
    if len(config['data']['test']['data']) == 0:
        print("测试集为空！")
    else:
        print(f"测试集大小：{len(config['data']['test']['data'])}")
    print()

    # 数据增强检查
    if config['data']['train']['Use_dataenhance']:
        print("训练集启用数据增强")
    else:
        print("训练集禁用数据增强")
    if config['data']['val']['Use_dataenhance'] or config['data']['test']['Use_dataenhance']:
        raise ValueError("验证集和测试集不应启用数据增强！")
    if config['data']['val']['data_per_picture'] > 1 or config['data']['test']['data_per_picture'] > 1:
        raise ValueError("验证集和测试集不应增强数据！")

    # batchsize检查
    print(f"the true batchsize = {config['data']['train']['dataloader_args']['batch_size'] * config['training']['accumulation_steps']}\
        because of the accumulation_steps = {config['training']['accumulation_steps']}")
