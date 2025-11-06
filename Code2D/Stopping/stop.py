from.manager import EarlyStopper

def get_manager(es_config: dict)-> EarlyStopper:
    '''
    Get the manager instance based on the config.
    '''
    manager = EarlyStopper(es_config)

    return manager


