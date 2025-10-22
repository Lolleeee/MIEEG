import torch
import numpy as np
from packages.plotting.reconstruction_plots import plot_reconstruction_slices, plot_reconstruction_scatter, plot_reconstruction_distribution

def autoencoder_test_plots(model, loader, nsamples=5):
    device = next(model.parameters()).device
    outputs = torch.tensor([]).to(device)
    
    for inputs in loader:
        print(inputs.shape)
        inputs = torch.tensor(inputs).to(device)
        if inputs.shape[0] > nsamples:
            inputs = inputs[:nsamples, ...]
            outputs = model(inputs)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            outputs = torch.cat((outputs, outputs), dim=0)
            break
        else:
            output = model(inputs)
            if isinstance(output, (list, tuple)):
                output = output[0]
            nsamples -= inputs.shape[0]
            outputs = torch.cat((outputs, output), dim=0)
            if nsamples <= 0:
                break
    rand_idx = np.random.randint(0, inputs.shape[0])
    single_input = inputs[rand_idx, ...].unsqueeze(0)
    single_output = model(single_input)
    if isinstance(single_output, (list, tuple)):
        single_output = single_output[0]
        
    plot_reconstruction_scatter(inputs, outputs)
    plot_reconstruction_distribution(inputs, outputs)

    plot_reconstruction_slices(single_input, single_output)

if __name__ == "__main__":
    from packages.data_objects.dataset import TorchDataset, CustomTestDataset
    from packages.io.file_loader import get_data_loaders 
    from packages.models.autoencoder_skip import Conv3DAE

    dataset_path = '/home/lolly/Projects/MIEEG/data/kaggle_eeg/preprocessed/'

    dataset = CustomTestDataset(nsamples=5, shape=(25, 7, 5, 64))

    train_loader, val_loader, _ = get_data_loaders(dataset, sets_size={'train': 0.7, 'val': 0.3, 'test': 0.}, norm_axes=(0, 4), batch_size = 128)

    model = Conv3DAE(
            use_convnext=True, 
            drop_p=0.1, 
            hidden_dims=[64, 128, 256],
            encoder_skip_connections=[(0, 2)],
            decoder_skip_connections=[(0, 2)],
            latent_dim=512
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    autoencoder_test_plots(model, val_loader, nsamples=5)