import yaml
import sys
import logging as log
import os

def parse(fp):
    with open(fp, 'r') as fd:
        cont = fd.read()
        y = yaml.load(cont)
        return y

def getConfig(cfgs_root, model_name):
    #cfgs_root = 'cfgs'
    main_cfg = parse('%s/main.yml' % cfgs_root)
    #model_name = main_cfg['model']
    if model_name not in main_cfg['cfg_dict'].keys():
        models = ', '.join(main_cfg['cfg_dict'].keys())
        print('There are models like %s\n' % models, file=sys.stderr)
        raise Exception
    cfg_fp = './' + cfgs_root + '/' + main_cfg['cfg_dict'][model_name]
    config =  parse(cfg_fp)
    #config['model_name'] = model_name
    #print(config)
    return config
