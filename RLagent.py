# Import Helpers
import os
import sys
import numpy as np
import math
import random
import time
import csv
import copy
import ctypes
from   os       import path
from   ctypes   import *
from   datetime import datetime

# Import Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim
import torch.multiprocessing as mp
from   torch import multinomial as multi
from   torch.distributions import Categorical
from   torch.optim import lr_scheduler

# Import GYM
from gym import Env

## Hyperparameters
n_train_processes = 1
learning_rate     = 0.0003
update_interval   = 8    # batch size
gamma             = 0.98
beta              = 1.0     # value loss coeff.
eta               = 0.0     # entropy loss coeff.
clip_grad         = 0.1
max_train_ep      = 200
featureNum        = 9
featureDim        = featureNum-2
networkDim        = 128
epsilon           = np.finfo(np.float32).eps.item()

## CUDA Device
os.environ["CUDA_VISIBLE_DEVICES"]='0'

## CSV File
curTime = datetime.now().strftime('%m%d_%H:%M:%S')
csvdir  = './csv/'
curCSV  = csvdir + curTime

## Trained Model
modeldir     = './model3/'
TrainedModel = modeldir + 'rldp_MODEL_0521_01:31:xx.pth'

## Gloabl variable for z-score normalization
SUM_mean  = 0
SUM_stdev = 10e-7
rewardNum = 0

#torch.set_printoptions(profile="full")

## Functions
def featuretoTensor(feature, device):
    s = copy.deepcopy(feature)
    # remove first two column for tensor
    for j in s:
        del j[:2]
    return torch.tensor(s, dtype=torch.float, device=device)

## Ctypes function
class circuit(object):
    ##  Legalization functions ##
    def __init__(self):
        self.lib = cdll.LoadLibrary('./object/libckt.so')
        self.obj = self.lib.ckt_new()
        self.features = []
    def parse(self, a):
        LP_c_char = POINTER(c_char)
        LP_LP_c_char = POINTER(LP_c_char)
        self.lib.argtypes = (c_int, LP_LP_c_char)
        argc = len(a)
        argv = (LP_c_char * (argc +1))()
        for i, arg in enumerate(a):
            arg     = arg.encode('utf-8')
            argv[i] = create_string_buffer(arg)
        self.lib.ckt_read_files(self.obj, argc, argv)
    def placeinit(self):
        self.lib.ckt_region_assn.restype = ctypes.c_int
        gNum = self.lib.ckt_region_assn(self.obj)
        return gNum
    def rtreeinit(self):
        self.lib.ckt_rtree_init(self.obj)
    #def place(self, gcell):
    def place(self):
        #self.lib.ckt_simple_placement(self.obj, gcell)
        self.lib.ckt_simple_placement(self.obj)
    def agent_clear(self, agent):
        self.lib.ckt_agent_clear.argtypes = [c_void_p, c_void_p]
        self.lib.ckt_agent_clear.restype = ctypes.c_double
        gScore = self.lib.ckt_agent_clear(self.obj, agent)
        return gScore
    def memory_clear(self, agent):
        self.lib.ckt_memory_clear.argtypes = [c_void_p, c_void_p]
        self.lib.ckt_memory_clear(self.obj, agent)
    def write(self):
        self.lib.ckt_write_def(self.obj)
    ##  RL-Agent functions ##
    def agent(self):
        agent = self.lib.agent_new()
        return agent
    def rl_init(self, agent, gcell):
        self.features = []
        self.lib.ckt_state_init.argtypes = [c_void_p, c_void_p, c_int]
        self.lib.ckt_state_init.restype = ctypes.c_int
        cellNum = self.lib.ckt_state_init(self.obj, agent, gcell)
        for i in range(cellNum):
            cell = []
            for j in range(featureNum):
                f = circuit.f_get(self, agent, i, j)
                cell.append(f)
            self.features.append(cell)
        return self.features
    def ep_done(self, agent):
        self.lib.ckt_is_done.restype = ctypes.c_bool
        isDone = self.lib.ckt_is_done(agent)
        return isDone
    def action(self, agent, act):
        self.lib.ckt_action.restype = ctypes.c_int
        self.lib.ckt_action.argtypes = [c_void_p, c_void_p, c_int]
        moveType = self.lib.ckt_action(self.obj, agent, act)
        return moveType
    def feature_update(self, agent, tarID, moveType, s_candi):
        self.lib.effected_cell_sidx.restype = ctypes.c_int
        self.lib.effected_cell_id.restype   = ctypes.c_int
        self.lib.ckt_feature_update.argtypes = [c_void_p, c_void_p, c_int, c_int]
        self.lib.ckt_feature_update(self.obj, agent, tarID, moveType)
        ecellNum = circuit.ecell_num(self, agent)
        ecellsidx = []
        ecellid = []
        for i in range(ecellNum):
            ecellsidx.append(self.lib.effected_cell_sidx(agent, i))
            ecellid.append(self.lib.effected_cell_id(agent, i))
        ecells = [ecellsidx, ecellid]
        # Update s_candi #
        for idxst, st in enumerate(s_candi):
            for idxec, ecid in enumerate(ecells[1]):
                if ecid == st[0]:
                    for kk in range(featureNum):
                        s_candi[idxst][kk] = circuit.f_get(self, agent, ecells[0][idxec], kk)
    def ecell_num(self, agent):
        self.lib.effected_cell_num.restype = ctypes.c_int
        ecellNum = self.lib.effected_cell_num(agent)
        return ecellNum
    def f_get(self, agent, cellidx, fidx):
        self.lib.feature_get.restype = ctypes.c_double
        f = self.lib.feature_get(agent, cellidx, fidx)
        return f
    def reward(self, agent):
        self.lib.ckt_reward_calc.restype = ctypes.c_double
        rew = self.lib.ckt_reward_calc(self.obj, agent)
        return rew
    def __del__(self):
        del self.lib
        del self.obj
        del self.features
        del self

## Environment
class RLegalizer(Env):
    def __init__(self):
        super(RLegalizer, self).__init__()
        self.ck = circuit()
        self.ck.parse(sys.argv)
        self.ag = self.ck.agent()
        self.gcNum = self.ck.placeinit()
        self.ck.place()

    def envinit(self):
        self.ck.rtreeinit()

    def s_init(self, gcell):
        #self.ck.place(gcell)
        return self.ck.rl_init(self.ag, gcell) 

    def done(self):
        return self.ck.ep_done(self.ag)

    def step(self, act, s_candi, info):
        moveType = self.ck.action(self.ag, act) # 0: move fail, 1: map move, 2: shift move
        self.ck.feature_update(self.ag, act, moveType, s_candi)
        reward = self.ck.reward(self.ag)
        done   = self.ck.ep_done(self.ag)
        info.append(moveType)
        return reward, done, info

    def agentclear(self):
        return self.ck.agent_clear(self.ag)     # After sub-episode (gcell)

    def memclear(self):
        self.ck.memory_clear(self.ag)           # After episode

    def write(self):
        self.ck.write()

    def __del__(self):
        del self.ck
        del self.ag
        del self

## Model
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1   = nn.Linear(featureDim, networkDim)
        self.fc2   = nn.Linear(networkDim, networkDim)
        self.fc_pi  = nn.Linear(networkDim, 1)
        self.fc_v   = nn.Linear(networkDim, 1)

        # layer initialization
        nn.init.uniform_(self.fc_v.weight, a=0, b=0.3)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc_pi.bias)
        nn.init.zeros_(self.fc_v.bias)
        
    def norm(self, x, outDim, isBatch):
        if isBatch == 0:
            x = F.normalize(x.reshape(-1, outDim), dim=0)
        else:
            # batch normalization (for columns)
            x = F.normalize(x.reshape(-1, outDim), dim=0).reshape(len(x),len(x[0]),outDim)
        return x

    def pi(self, x, softmax_dim=0):
        x = self.norm(x, featureDim, softmax_dim) + epsilon
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim) + epsilon
        return prob

    def v(self, x, norm_dim=0):
        x = self.norm(x, featureDim, norm_dim) + epsilon
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_v(x)
        return x.mean(dim=norm_dim)


## Training
def train(global_model, rank, device):

    ## Initialize the environment
    env         = RLegalizer()
    ## Define the device, model, and optimizer
    device      = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    global_model.to(device)
    local_model = ActorCritic().to(device)
    local_model.load_state_dict(global_model.state_dict())
    #optimizer   = optim.RMSprop(global_model.parameters(), lr=learning_rate, centered=False)
    optimizer   = optim.Adam(global_model.parameters(), lr=learning_rate)
    #scheduler   = lr_scheduler.ExponentialLR(optimizer, gamma=0.97) #not worked
    
    start_time = time.time()
    ## Perform the entire circuit DP 'max_train_ep' times
    for n_epi in range(max_train_ep):
        print("[TRAIN-{}] EPISODE #{}".format(rank, n_epi))
        csv_y = []
        # Reset environment
        env.envinit()

        ## Perform sub-episode for gcells
        #for gcell in range(env.gcNum):         ## This is for whole circuit
        for kk in range(1):
            gcell=1

            ## Initialize
            s       = env.s_init(gcell)     # state type: 2D-list
            #print("s: {}".format(s))
            done    = env.done()
            r       = 0
            stepN   = 0
            s_candi = copy.deepcopy(s)

            ## 4. Do 2 ~ 3 until DONE
            while not done:
                s_lst, r_lst = [], []
                
                ## 2. As much as batch size
                for t in range(update_interval):
                    stepN+=1
                    
                    ## (1) Select actions
                    s_can_t = featuretoTensor(s_candi, device)
                    prob    = local_model.pi(s_can_t)
                    #print("prob: {}".format(prob))
                    if torch.isnan(prob).any():
                        print("state: {}".format(s_candi))
                        print("prob: {}".format(prob))
                    probf   = prob.flatten()
                    act = Categorical(probf).sample().item()
                    
                    ## Exit program if the placed cell is selected
                    if s_candi[act][1] == 1.0:
                        print("[ERROR] [TRAIN-{}] Tried Cell is selected AGAIN ({}-th cell: {})".format(rank, act, int(s_candi[act][0])))
                        sys.exit()
                    
                    ## (3) Collect parameters
                    s_lst.append(s_can_t)
                    ## (2) Do step
                    info   = []
                    r, done, info = env.step(int(s_candi[act][0]), s_candi, info)
                    
                    ## (3) Collect parameters
                    if t == 0:
                        pi_a_tens  = prob[act]
                        pi_entropy = (prob * torch.log(prob)).sum(dim=0)
                    else:
                        pi_a_tens  = torch.cat([pi_a_tens, prob[act]], dim=0)
                        pi_entropy = torch.cat([pi_entropy, (prob * torch.log(prob+epsilon)).sum(dim=0)], dim=0)

                    if info[0] == 1:
                        r_lst.append(1.0*r)
                    else:
                        r_lst.append(0.0)
                    print("[TRAIN-{}] Step-{} reward: {:.4f}".format(rank, stepN, r_lst[len(r_lst)-1]))
                    if done:
                        break
                    
                    ## (4) State update (s_candi): remove tried cells
                    if info[0] == 1:
                        del s_candi[act]
                    else:
                        del s_candi[act]
                        for index, cell in reversed(list(enumerate(s_candi))):
                            if cell[1] == 1.0:
                                del s_candi[index]

                ## Skip backpropagation for below-5-cell Gcells
                if len(s) < 10:
                    continue
                ## Generate Q value (td_target)
                s_tens  = featuretoTensor(s_candi, device)
                s_final = s_tens.clone().detach()
                R = 0.0 if done else local_model.v(s_final).item()
                print("R: {}".format(R))
                td_target_lst = []
                for reward in r_lst[::-1]:
                    R = gamma * R + reward
                    td_target_lst.append([R])
                td_target_lst.reverse()
                td_target = torch.tensor(td_target_lst, dtype=torch.float, device=device)

                ## Generate V value (v_batch)
                for scidx, sc in enumerate(s_lst):
                    if scidx == 0:
                        v_batch = local_model.v(sc)
                    else:
                        v_batch = torch.cat([v_batch, local_model.v(sc)], dim=0)

                v_batch    = v_batch.unsqueeze(-1)
                pi_a_tens  = pi_a_tens.unsqueeze(-1)
                pi_entropy = pi_entropy.unsqueeze(-1)

                ## Advantage function (A = Q-V)
                advantage = td_target - v_batch
                
                ## loss funcitons ##
                policy_loss  = -torch.log(pi_a_tens) * advantage.detach()
                print("pLoss: {}".format(policy_loss.mean()))

                value_loss   = F.smooth_l1_loss(v_batch, td_target.detach())
                print("vLoss: {}".format(value_loss.mean()))
                
                entropy_loss = pi_entropy

                loss = policy_loss + beta * value_loss + eta * entropy_loss
    
                ## Take smaller learning rate for small batch size (at the last of sub-epi)
                #optimizer   = optim.Adam(global_model.parameters(), lr=lrate)
                optimizer.zero_grad()
                loss.mean().backward()
                
                ## clip grad
                nn_utils.clip_grad_norm_(local_model.parameters(), clip_grad)
                
                for global_param, local_param in zip(global_model.parameters(),local_model.parameters()):
                    global_param._grad = local_param.grad

                optimizer.step()
                
                #scheduler.step()
                local_model.load_state_dict(global_model.state_dict())
            
            ## After a sub-episode done, get score and append in csv file
            gcScore = env.agentclear()
            csv_y.append(gcScore)
            print("[TRAIN-{}] GCELL#{} END (Score: {})".format(rank, gcell, gcScore))

        print("[TRAIN-{}] EPISODE#{} END".format(rank, n_epi))
        ## After a episode done, clear memory.
        if n_epi != max_train_ep-1:
            env.memclear()

        ## Model save for every episode
        torch.save(global_model, modeldir+'rldp_MODEL_'+curTime+'.pth')

        ## Write CSV score file
        w  = open(curCSV+'-rank{}'.format(rank)+'.csv', 'a')
        wr = csv.writer(w)
        for csvidx, cs in enumerate(csv_y):
            wr.writerow([csvidx, n_epi, cs])
        w.close()

    ## Write final placed DEF file and close environment 
    env.write()
    env.close()
    del env
    print("[TRAIN] Training process {} reach maximum episode.".format(rank))
    print("Time laps:", (time.time() - start_time) / 60, "min")



## Main

if __name__ == '__main__':

    ## Print hyperparameters
    print("--- Hyperparameters ---")
    print(" num_agent: {}".format(n_train_processes))
    print(" lr: {}".format(learning_rate))
    print(" batch size: {}".format(update_interval))
    print(" gamma: {}".format(gamma))
    print(" beta: {}".format(beta))
    print(" eta: {}".format(eta))
    print(" clip_grad: {}".format(clip_grad))
    print(" # of epi: {}".format(max_train_ep))
    print(" feature dim: {}".format(featureDim))
    print(" network dim: {}".format(networkDim))
    print("-----------------------")
    
    ## First, load model in 'cpu' device in main(), 
    ## then in each process, we map the model in 'cuda'
    device = torch.device('cpu')

    if path.exists(TrainedModel):
        print("Trained Model: {}".format(TrainedModel))
        global_model = torch.load(TrainedModel, map_location=device)
    else:
        global_model = ActorCritic().to(device)
    global_model.share_memory()
    global_model.train()

    # Multi-process training for multi-agents
    processes = []
    for rank in range(n_train_processes):
        global_model.train()
        p = mp.Process(target=train, args=(global_model, rank, device))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print("-- Program End! --")

### END PROGRAM ###
