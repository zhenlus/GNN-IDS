import numpy as np
import torch
import os
import random

def set_seed(seed: int = 40) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")

def benign_data(num_samples, num_nodes, action_nodes, rt_meas_dim):
    """ Generate benign data
    """
    set_seed()
    action_node_idx = list(action_nodes.keys())
    num_action_nodes = len(action_node_idx)

    X = torch.zeros(num_samples, num_nodes, rt_meas_dim)
    Y = torch.zeros(num_samples, num_nodes, dtype=torch.float32)
    sd = 0.2
    rt_measurements = []
    for i in range(rt_meas_dim//3):
        mu = np.random.uniform(0.3, 0.3)
        lambda_p = np.random.uniform(3.0, 3.0)
        rt_1 = torch.normal(mu, sd, size=(num_samples, num_action_nodes))
        rt_2 = torch.poisson(torch.ones(num_samples, num_action_nodes)*lambda_p)
        rt_3 = rt_1.abs() ** 0.5 + rt_2 * .5 + 0.5
        rt_measurements.append(torch.stack((rt_1, rt_2, rt_3), dim=2))
    
    rt_measurements = torch.cat(rt_measurements, dim=2)
    X[:, action_node_idx, :] = rt_measurements

    return X, Y

def malic_data(num_samples, num_nodes, action_nodes, rt_meas_dim):
    """ Generate malicious data
    """
    action_node_idx = list(action_nodes.keys())

    comp_node = [[i] for i in action_node_idx]
    num_comp_scenarios = len(comp_node)
    X = torch.zeros(num_samples*num_comp_scenarios, num_nodes, rt_meas_dim)
    Y = torch.zeros(num_samples*num_comp_scenarios, num_nodes, dtype=torch.float32)

    for idx, scenario in enumerate(comp_node):
        num_comp_nodes = len(scenario)
        x, y = benign_data(num_samples, num_nodes, action_nodes, rt_meas_dim)
        action_name = action_nodes[scenario[0]]['predicate']

        mali_meas = sample_mali_scen(num_samples, num_comp_nodes, rt_meas_dim, action_name)

        x[:, scenario, :] = mali_meas
        y[:, scenario] = 1

        X[idx*num_samples:(idx+1)*num_samples, :, :] = x
        Y[idx*num_samples:(idx+1)*num_samples, :] = y

    return X, Y


def sample_mali_scen(num_samples, num_comp_nodes, rt_meas_dim, action_name):
    """ Sample malicious measurements for a scenario
    """
    set_seed()
    rt_measurements = []
    if 'access' in action_name.lower():
        sd = 0.2
        for i in range(rt_meas_dim//3):
            mu = np.random.uniform(0.3, 0.3)
            lambda_p = np.random.uniform(1, 5)
            rt_1 = torch.normal(mu, sd, size=(num_samples, num_comp_nodes))
            rt_2 = torch.poisson(torch.ones(num_samples, num_comp_nodes)*lambda_p)
            rt_3 = rt_1.abs() ** 0.5 + rt_2 * .5 + 0.5
            rt_measurements.append(torch.stack((rt_1, rt_2, rt_3), dim=2))

    elif 'execcode' in action_name.lower():
        sd = 0.1
        for i in range(rt_meas_dim//3):
            mu = np.random.uniform(0.1, 0.5)
            lambda_p = np.random.uniform(3.0, 3.0)
            rt_1 = torch.normal(mu, sd, size=(num_samples, num_comp_nodes))
            rt_2 = torch.poisson(torch.ones(num_samples, num_comp_nodes)*lambda_p)
            rt_3 = rt_1.abs() ** 0.5 + rt_2 * .5 + 0.5

            rt_measurements.append(torch.stack((rt_1, rt_2, rt_3), dim=2))
    else:
        raise ValueError('event_type should be one of access, execute')
    return torch.cat(rt_measurements, dim=2)

def gene_dataset(num_benign, num_malic, num_nodes, action_nodes, rt_meas_dim):
    """ Generate  Dataset 1
    """
    X_benign, Y_benign = benign_data(num_benign, num_nodes, action_nodes, rt_meas_dim)
    X_malic, Y_malic   = malic_data(num_malic, num_nodes, action_nodes, rt_meas_dim)

    action_mask = list(action_nodes.keys())

    X = torch.cat((X_benign, X_malic), dim=0)
    Y = torch.cat((Y_benign, Y_malic), dim=0)[:, action_mask]

    return X, Y