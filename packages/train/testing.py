import torch
import numpy as np
from packages.plotting.reconstruction_plots import plot_reconstruction_slices, plot_reconstruction_scatter, plot_reconstruction_distribution

def autoencoder_test_plots(model, loader, nsamples=5):
    device = next(model.parameters()).device
    outputs = torch.tensor([]).to(device)
    
    for inputs in loader:
        inputs = torch.tensor(inputs).to(device)
        if inputs.shape[0] > nsamples:
            inputs = inputs[:nsamples, ...]
            outputs = model(inputs)
            outputs = torch.cat((outputs, outputs), dim=0)
            break
        else:
            output = model(inputs)
            nsamples -= inputs.shape[0]
            outputs = torch.cat((outputs, output), dim=0)
            if nsamples <= 0:
                break
    rand_idx = np.random.randint(0, inputs.shape[0])
    single_input = inputs[rand_idx, ...]
    single_output = model(single_input)
    
    plot_reconstruction_scatter(inputs, outputs)
    plot_reconstruction_distribution(inputs, outputs)

    plot_reconstruction_slices(single_input, single_output)