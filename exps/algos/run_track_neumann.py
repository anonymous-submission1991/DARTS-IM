import os, sys, time, glob, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from pathlib import Path
sys.path.append(str('/home/mzhang3/Data/2022ICML/NAS_Bench201/lib'))
sys.path.append(str('/home/mzhang3/Data/2022ICML/NAS_Bench201/configs'))
from config_utils import load_config, dict2config, configure2str
from datasets     import get_datasets, SearchDataset
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils        import get_model_infos, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time
from models       import get_cell_based_tiny_net, get_search_spaces
from nas_102_api  import NASBench102API as API

from weight_initializers import init_net
def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


def _hessian_vector_product_alpha(vector, network, criterion, base_inputs, base_targets, r=1e-2): ###calculate vector * /patial L^2 / /patial /alpha /patial /theta 
    
    base_targets = base_targets.cuda(non_blocking=True)     
    R = r / _concat(vector).norm()
    for p, v in zip(network.module.get_alphas(), vector):
        p.data.add_(R*v)
    _, logits = network(base_inputs)
    loss = criterion(logits, base_targets)
    grads_p = torch.autograd.grad(loss, network.module.get_weights())

    for p, v in zip(network.module.get_alphas(), vector):
        p.data.sub_(2*R*v)
    _, logits = network(base_inputs)
    loss = criterion(logits, base_targets)
    grads_n = torch.autograd.grad(loss, network.module.get_weights())

    for p, v in zip(network.module.get_alphas(), vector):
        p.data.add_(R*v)
    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]


def _hessian_vector_product_weight(vector, network, criterion, base_inputs, base_targets, r=1e-2): ###calculate vector * /patial L^2 / /patial /alpha /patial /theta 
    
    base_targets = base_targets.cuda(non_blocking=True)     
    R = r / _concat(vector).norm()
    for p, v in zip(network.module.get_weights(), vector):
        p.data.add_(R*v)
    _, logits = network(base_inputs)
    loss = criterion(logits, base_targets)
    grads_p = torch.autograd.grad(loss, network.module.get_alphas())

    for p, v in zip(network.module.get_weights(), vector):
        p.data.sub_(2*R*v)
    _, logits = network(base_inputs)
    loss = criterion(logits, base_targets)
    grads_n = torch.autograd.grad(loss, network.module.get_alphas())

    for p, v in zip(network.module.get_weights(), vector):
        p.data.add_(R*v)
    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]



def _vector_hessian_product_neumann_3(vector, network, criterion, base_inputs, base_targets, eta):
    save_model = deepcopy(network)       

    p_model = deepcopy(network)       
    base_targets = base_targets.cuda(non_blocking=True)        
    n = base_inputs.size(0)         
    _, logits = p_model(base_inputs)      
    loss = criterion(logits, base_targets)      
    grads = torch.autograd.grad(loss, p_model.module.get_weights(),create_graph=True)   
    prod=sum([(g*v).sum() for g,v in zip(grads, vector)]) 
    prod.backward(retain_graph=True)        

    grads_1=[v-eta*p.grad.data for v,p in zip(vector, p_model.module.get_weights())] 

    del p_model
    p_model = deepcopy(network)       
    base_targets = base_targets.cuda(non_blocking=True)        
    n = base_inputs.size(0)         
    _, logits = p_model(base_inputs)      
    loss = criterion(logits, base_targets)      
    grads = torch.autograd.grad(loss, p_model.module.get_weights(),create_graph=True)   
    prod=sum([(g*v).sum() for g,v in zip(grads, grads_1)]) 
    prod.backward(retain_graph=True)     

    grads_2=[v-eta*p.grad.data for v,p in zip(grads_1, p_model.module.get_weights())] 

    del p_model
    p_model = deepcopy(network)       
    base_targets = base_targets.cuda(non_blocking=True)        
    n = base_inputs.size(0)         
    _, logits = p_model(base_inputs)      
    loss = criterion(logits, base_targets)      
    grads = torch.autograd.grad(loss, p_model.module.get_weights(),create_graph=True)   
    prod=sum([(g*v).sum() for g,v in zip(grads, grads_2)]) 
    prod.backward(retain_graph=True)     

    grads_3=[v-eta*p.grad.data for v,p in zip(grads_2, p_model.module.get_weights())]     

    hessian_vector=[v.data+p.data+g.data+kk.data for v,p,g,kk in zip(vector, grads_1, grads_2, grads_3)] 
 
    return hessian_vector



def _operation_importance(network, criterion, base_inputs, base_targets, eta):
    
    alpha_ones= [v+torch.ones_like(v)-v.data for v in network.module.get_alphas()]  
    alpha_ones_H= _hessian_vector_product_alpha(alpha_ones, network, criterion, base_inputs, base_targets, r=1e-2)
    
    alpha_ones_H_H_1= _vector_hessian_product_neumann_3(alpha_ones_H, network, criterion, base_inputs, base_targets, eta=1e-3)

    alpha_ones_H_H_1_H = _hessian_vector_product_weight(alpha_ones_H_H_1, network, criterion, base_inputs, base_targets, r=1e-2)
    
    return alpha_ones_H_H_1_H
    
    
parser = argparse.ArgumentParser("DARTS Second Order")
parser.add_argument('--data_path',          type=str,   default= '/data/mzhang3/ENNAS_master/data/cifar-10-batches-py', help='Path to dataset')
parser.add_argument('--dataset',            type=str,   default= 'cifar10', choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
# channels and number-of-cells
parser.add_argument('--search_space_name',  type=str,   default= 'nas-bench-102',help='The search space name.')
parser.add_argument('--max_nodes',          type=int,   default= 4 ,help='The maximum number of nodes.')
parser.add_argument('--channel',            type=int,   default= 16,help='The number of channels.')
parser.add_argument('--num_cells',          type=int,   default= 5, help='The number of cells in one stage.')
# architecture leraning rate
parser.add_argument('--arch_learning_rate', type=float, default=3e-2, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay',  type=float, default=1e-3, help='weight decay for arch encoding')
# log
parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
parser.add_argument('--save_dir',           type=str,   default='/output/search-cell-nas-bench-102-cifar10',help='Folder to save checkpoints and log.')
parser.add_argument('--arch_nas_dataset',   type=str,   default='/data/mzhang3/2022ICML/NAS_Bench201/NAS-Bench-201-v1_1-096897.pth',help='The path to load the architecture dataset (tiny-nas-benchmark).')
parser.add_argument('--print_freq',         type=int,   default=50,help='print frequency (default: 200)')
parser.add_argument('--rand_seed',          type=int,   default=0, help='manual seed')
parser.add_argument('--batches',          type=int,   default=1, help='batch number')
args = parser.parse_args()



args.save_dir=args.save_dir= './ZZZZ_INTER_RESULT/run_track_example_batches_neumann_'+str(args.rand_seed)+str(args.batches)+args.save_dir

       
xargs=args
assert torch.cuda.is_available(), 'CUDA is not available.'
torch.cuda.set_device(0)
torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_num_threads( xargs.workers )
prepare_seed(xargs.rand_seed)
logger = prepare_logger(args)

train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
if xargs.dataset == 'cifar10' or xargs.dataset == 'cifar100':
    split_Fpath = '/home/mzhang3/Data/2021NeurIPS_codes/NAS_Bench201/configs/nas-benchmark/cifar-split.txt'
    cifar_split = load_config(split_Fpath, None, None)
    train_split, valid_split = cifar_split.train, cifar_split.valid
    logger.log('Load split file from {:}'.format(split_Fpath))
elif xargs.dataset.startswith('ImageNet16'):
    split_Fpath = 'configs/nas-benchmark/{:}-split.txt'.format(xargs.dataset)
    imagenet16_split = load_config(split_Fpath, None, None)
    train_split, valid_split = imagenet16_split.train, imagenet16_split.valid
    logger.log('Load split file from {:}'.format(split_Fpath))
else:
    raise ValueError('invalid dataset : {:}'.format(xargs.dataset))
config_path = '/home/mzhang3/Data/2021NeurIPS_codes/NAS_Bench201/configs/nas-benchmark/algos/DARTS.config'
config = load_config(config_path, {'class_num': class_num, 'xshape': xshape}, logger)
# To split data
train_data_v2 = deepcopy(train_data)
train_data_v2.transform = valid_data.transform
valid_data    = train_data_v2
search_data   = SearchDataset(xargs.dataset, train_data, train_split, valid_split)
# data loader
search_loader = torch.utils.data.DataLoader(search_data, batch_size=config.batch_size, shuffle=True , num_workers=xargs.workers, pin_memory=True)
valid_loader  = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split), num_workers=xargs.workers, pin_memory=True)
logger.log('||||||| {:10s} ||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(search_loader), len(valid_loader), config.batch_size))
logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

search_space = get_search_spaces('cell', xargs.search_space_name)
model_config = dict2config({'name': 'DARTS-V2', 'C': xargs.channel, 'N': xargs.num_cells,
                              'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                              'space'    : search_space}, None)
search_model = get_cell_based_tiny_net(model_config)
logger.log('search-model :\n{:}'.format(search_model))

w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.get_weights(), config)
a_optimizer = torch.optim.Adam(search_model.get_alphas(), lr=xargs.arch_learning_rate, betas=(0.5, 0.999), weight_decay=xargs.arch_weight_decay)


if xargs.arch_nas_dataset is None:
    api = None
else:
    api = API(xargs.arch_nas_dataset)
logger.log('{:} create API = {:} done'.format(time_string(), api))

last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()    


for i in range(50):
    epo=i+1
    pa='/home/mzhang3/Data/2022ICML/NAS_Bench201/exps/algos/DARTS_track_V2_seed'+str(args.rand_seed)+'/output/search-cell-nas-bench-102-cifar10/checkpoint/seed-'+str(args.rand_seed)+'-basic-epoch-'+str(epo)+'.pth'
    checkpoint  = torch.load(pa)
    valid_accuracies = checkpoint['valid_accuracies']
    search_model.load_state_dict( checkpoint['search_model'] )
    w_scheduler.load_state_dict ( checkpoint['w_scheduler'] )
    w_optimizer.load_state_dict ( checkpoint['w_optimizer'] )
    a_optimizer.load_state_dict ( checkpoint['a_optimizer'] )

    network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda() 
    logger.log('--------------------------track epoch '+str(epo)+'--------------------------')
    save_model = deepcopy(network)
    ori_arch=deepcopy(network.module.get_alphas()[0].data)
    for i in range(6):
        ori_arch[i,0]=float("-Inf")
    logger.log('{:}'.format(api.query_by_arch(search_model.genotype_prune(ori_arch)))) 
    #network.module.arch_parameters.data[0,1]=float("-Inf")
    weight_decay=w_optimizer.param_groups[0]['weight_decay']

    N=args.batches
    operation_importance=torch.zeros_like(network.module.arch_parameters)
    for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(search_loader):  
        network.zero_grad()
        arch_inputs = arch_inputs.cuda(non_blocking=True)    
        arch_targets = arch_targets.cuda(non_blocking=True)       

        operation_importance=operation_importance+_operation_importance(network, criterion, arch_inputs, arch_targets, eta=1e-3)[0]
        #print(vector_hessian_vector_losses.avg)   
        if step>N-2:
            break
    operation_importance=operation_importance/(step+1)

    alpha_data=network.module.get_alphas()[0].data
    alpha_2=alpha_data.pow(2)

    operation_importance_final= -operation_importance

    #logger.log('--------------------------below IM----------------------')
    #logger.log('{:}'.format(api.query_by_arch(search_model.genotype_prune(operation_importance_final))))     

    for i in range(6):######we need to follow DARTS to exclude none operation. You could also reserve it.
        operation_importance_final[i,0]=float("-Inf")    

    logger.log('--------------------------below IM with out none----------------------')
    logger.log('{:}'.format(api.query_by_arch(search_model.genotype_prune(operation_importance_final))))   


