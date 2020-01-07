import os

def check_dir(dirname, report_error=False):
    """ Check existence of specific directory
    Args:
        dirname: path to specific directory
        report_error: if it is True, error will be report when dirname not exists
                      if it is False, directory will be created when dirnam not exists
    Return:
        None
    Raise:
        ValueError, if report_error is True and dirname not exists 
    """
    if not os.path.exists(dirname):
        if report_error is True:
            raise ValueError('not exist directory: {}'.format(dirname))
        else:
            os.makedirs(dirname)
            print('not exist {}, but has been created'.format(dirname))