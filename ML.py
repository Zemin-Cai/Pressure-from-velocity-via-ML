import torch
import torch.nn as nn
from torch import optim, autograd
from collections import OrderedDict
import time
import numpy as np
import scipy.io
import random
from torch import Tensor
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(1)

class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1
        self.activation = torch.nn.SiLU

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
        return u, v, p

def gradients(input, output):
    return autograd.grad(outputs=output, inputs=input,
                                grad_outputs=torch.ones_like(output),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

def sample_random(X_all, N):
    idx = np.random.choice(X_all.shape[0], N, replace=False)
    X_sampled = X_all[idx, :]
    return X_sampled

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

def error_l2(x, y):
    return torch.norm(x - y) / torch.norm(y)

def main():

    epochs = 100000
    N_data = 50000
    N_eqns = 50000
    Rey = 1
    layers = [3,100,100,100,100,100,100,100,100,3]

    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DNN(layers).to(device)
    model = nn.DataParallel(model, device_ids=[0, 1,2])
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.5)

    # print(model)

    data = scipy.io.loadmat(r'taylor.mat')
    t_st = data['T']
    T_star = t_st.flatten()[:, None]
    x_st = data['X']
    X_star = x_st.flatten()[:, None]
    y_st = data['Y']
    Y_star = y_st.flatten()[:, None]
    u_st = data['U']
    U_star = u_st.flatten()[:, None]
    v_st = data['V']
    V_star = v_st.flatten()[:, None]
    p_st = data['P']
    P_star = p_st.flatten()[:, None]

    train_data_all = np.hstack((T_star, X_star, Y_star, U_star, V_star, P_star))
    train_data_all1 = torch.from_numpy(train_data_all).float()
    train_data_all1 = train_data_all1.to(device)

    train_data = sample_random(train_data_all1, N_data)
    co_data = train_data[:,:3]
    exact_u = train_data[:,3:4]
    exact_v = train_data[:,4:5]

    train_eqns = sample_random(train_data_all1, N_eqns)
    eq_data = train_eqns[:,:3]
    t_eq_data = eq_data[:,0:1]
    x_eq_data = eq_data[:,1:2]
    y_eq_data = eq_data[:,2:3]

    tt = time.time()

    for epoch in range(epochs+1):

        output_u_data, output_v_data, output_p_data = model(co_data)
        loss_u = 1 * torch.mean(torch.abs(output_u_data - exact_u))
        loss_v = 1 * torch.mean(torch.abs(output_v_data - exact_v))

        # output_u_bo, output_v_bo, output_p_bo = model(bo_data)
        # loss_p = 1 * torch.mean(torch.abs(output_p_bo - exact_p))

        t_eq_data.requires_grad_()
        x_eq_data.requires_grad_()
        y_eq_data.requires_grad_()
        output_u_eqns, output_v_eqns, output_p_eqns = model(torch.hstack((t_eq_data, x_eq_data, y_eq_data)))
        u_t = gradients(t_eq_data, output_u_eqns)
        u_x = gradients(x_eq_data, output_u_eqns)
        u_xx = gradients(x_eq_data, u_x )
        u_y = gradients( y_eq_data, output_u_eqns)
        u_yy = gradients( y_eq_data, u_y)

        v_t = gradients(t_eq_data, output_v_eqns)
        v_x = gradients( x_eq_data, output_v_eqns)
        v_xx = gradients( x_eq_data, v_x)
        v_y = gradients(y_eq_data, output_v_eqns)
        v_yy = gradients(y_eq_data, v_y)

        p_x = gradients(x_eq_data, output_p_eqns)
        p_y = gradients(y_eq_data, output_p_eqns)


        e1 = u_t + output_u_eqns * u_x + output_v_eqns * u_y + p_x - (Rey) * (u_xx + u_yy)
        e2 = v_t + output_u_eqns * v_x + output_v_eqns * v_y + p_y - (Rey) * (v_xx + v_yy)

        loss_r1 = 1 * torch.mean(torch.abs( e1 ))
        loss_r2 = 1 * torch.mean(torch.abs( e2 ))


        # loss = loss_u + loss_v + loss_p + loss_r1 + loss_r2
        loss = loss_u + loss_v + loss_r1 + loss_r2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        StepLR.step()
        if epoch % 100 == 0:
            # print('epoch:', epoch, 'loss:', loss.item(), 'loss_u:',loss_u.item(), 'loss_v:', loss_v.item(), 'loss_p:', loss_p.item(), 'loss_r1:', loss_r1.item(), 'loss_r2:', loss_r2.item())
            print('epoch:', epoch, 'loss:', loss.item(), 'loss_u:',loss_u.item(), 'loss_v:', loss_v.item(), 'loss_r1:', loss_r1.item(), 'loss_r2:', loss_r2.item())

    print(time.time() - tt)

    u_pred, v_pred, p_pred = model(train_data_all1[:,:3])
    u_pred = u_pred.cpu().detach().numpy()
    v_pred = v_pred.cpu().detach().numpy()
    p_pred = p_pred.cpu().detach().numpy()


    error_u_relative = np.linalg.norm(train_data_all[:,3:4] - u_pred, 2) / np.linalg.norm(train_data_all[:,3:4], 2)
    error_u_abs = np.mean(np.abs(train_data_all[:,3:4] - u_pred))
    print('L2:', error_u_relative)
    print('L1:', error_u_abs)

    error_v_relative = np.linalg.norm(train_data_all[:, 4:5] - v_pred, 2) / np.linalg.norm(train_data_all[:, 4:5], 2)
    error_v_abs = np.mean(np.abs(train_data_all[:, 4:5] - v_pred))
    print('L2:', error_v_relative)
    print('L1:', error_v_abs)

    scipy.io.savemat('taylor2.mat' , {'u_pred':u_pred, 'v_pred':v_pred, 'p_pred':p_pred})


if __name__ == '__main__':
    main()