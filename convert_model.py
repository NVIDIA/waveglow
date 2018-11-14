import sys
import copy
import torch

if __name__ == '__main__':
    old_model_path = sys.argv[1]
    new_model_path = sys.argv[2]
    model = torch.load(old_model_path)
    old_model = model['model']
    new_model = copy.deepcopy(old_model)
    for idx in range(0, len(new_model.WN)):
        wavenet = new_model.WN[idx]
        wavenet.res_skip_layers = torch.nn.ModuleList()
        n_channels = wavenet.n_channels
        n_layers = wavenet.n_layers
        for i in range(0, n_layers):
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            skip_layer = wavenet.skip_layers[i]
            if i < n_layers - 1:
                res_layer = wavenet.res_layers[i]
                res_skip_layer.weight = torch.cat([res_layer.weight, skip_layer.weight])
            else:
                res_skip_layer.weight = skip_layer.weight
            wavenet.res_skip_layers.append(res_skip_layer)
        del wavenet.res_layers
        del wavenet.skip_layers
    model['model'] = new_model
    torch.save(model, new_model_path)
    
