# Objective
To rerun the notebooks and ensure they're compatible with the latest releases of dependant libraries.

# Notebooks

- TextCNN.ipynb

  Runs smoothly, nothing to highlight.

- VAE.ipynb

  Runs smoothly, nothing to highlight.

- FashionMNIST.ipynb

  ```python 
  # Issue in loading .pt file
  def fetch_last_checkpoint_model_filename(model_save_path):

    import os
    checkpoint_files = os.listdir(model_save_path)
    checkpoint_files = [f for f in checkpoint_files if '.pth' in f]
    checkpoint_iter = [int(x.split('_')[2].split('.')[0]) for x in checkpoint_files]
    last_idx = np.array(checkpoint_iter).argmax()
    return os.path.join(model_save_path, checkpoint_files[last_idx])

  model.load_state_dict(torch.load(fetch_last_checkpoint_model_filename('./saved_models')))
  print("Model Loaded")
  ```

  Changed to 

  ```python
  # loading the saved model
  def fetch_last_checkpoint_model_filename(model_save_path):

    import os
    checkpoint_files = os.listdir(model_save_path)
    checkpoint_files = [f for f in checkpoint_files if '.pt' in f]
    checkpoint_iter = [int(x.split('_')[2].split('.')[0]) for x in checkpoint_files]
    last_idx = np.array(checkpoint_iter).argmax()
    return os.path.join(model_save_path, checkpoint_files[last_idx])

  model.load_state_dict(torch.load(fetch_last_checkpoint_model_filename('./saved_models')))
  print("Model Loaded")
  ```

- CycleGAN_with_ignite_and_torch_cuda_amp.ipynb

  Runs smoothly, nothing to highlight.

- CycleGAN_with_ignite_and_nvdia_apex.ipynb

  Runs smoothly, nothing to highlight.

- FastaiLRFinder_MNIST.ipynb

  Added a couple of lines to the notebook to install ignite.

