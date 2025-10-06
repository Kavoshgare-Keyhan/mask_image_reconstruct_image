# Reconstruct Masked Images

In this project, we have used 100 classes of ImageNet and loaded a limited number of images from each class to apply generative episodic memory concept. 

Implementation Pipeline/Stages:

- Create VQ-VAE Module (**vqvae.py**): Set up a proper vqvae model and every methods that needed for your purposes
- Train the VQ-VAE (**train_vqvae.py**): Load images and then train your model until reach your desired loss and then save final model.
    - What is our desired loss (Overall loss, reconstruction loss, codebook loss, and commitment loss)? **A well-trained VQ-VAE should exhibit low reconstruction loss (Mean Squared Error, MSE, or Binary Cross-Entropy, BCE, depending on the data type).** In the end, we must minimize the total loss for test data.
    - **Must revise, because we need a test set and then apply on them to get optimum model for a specific epoch.** It is also possible to have a validation set to determine hyperparameter to have a well-trained model.
    - Advice to implement in the best way: 
        1. Split dataset
            - 70% train
            - 15% Validation
            - 15% Test
        2. Train the model on 70% of data
            - Monitor training loss over epochs
            - Use early stopping if validation loss starts increasing (to avoid overfitting)
        3. Hyperparameter uning of the Following Parameters
            - Codebook size
            - Embedding dimension
            - Commitment loss weight (β)
            - Learning rate
            - Batch size
            - Use *Bayesian Optimization* (custom code and need to set up by your own) or *Optuna*
        4. Select the best model
            - Retrain the model on the train + validation sets (optional but recommended)
            - Evaluate final performance on the test set
            - Report metrics: total loss, reconstruction loss, and optionally visual samples
        5. Practical Tips:
            - Loss Curves: Plot training and validation loss over epochs to diagnose underfitting or overfitting.
            - Model Checkpointing: Save the model with the lowest validation loss during training.
            - Seed Control: Set random seeds for reproducibility.
            - Normalization: Normalize input data consistently across all splits.
            - Use MLflow or Neptune to track the errors
    🚀 Running Instructions
        ✅ Use full config:
            bash
            torchrun --nproc_per_node=4 train.py --config config.yaml
        ✅ Override GPU selection:
            bash
            torchrun --nproc_per_node=4 train.py --config config.yaml --gpu_ids 1 3
        ✅ Filter by threshold:
            bash
            torchrun --nproc_per_node=4 train.py --config config.yaml --threshold 5.0 --filter_by_threshold
        ✅ Combine both:
            bash
            torchrun --nproc_per_node=4 train.py --config config.yaml --gpu_ids 1 3 --threshold 5.0 --filter_by_threshold
        ✅ Automatically detect available gpus below specific threshold
            torchrun --nproc_per_node=$(python -c "import yaml; cfg=yaml.safe_load(open('config.yaml')); from gpu_utils import select_gpus; print(len(select_gpus(cfg['multiprocessing']['gpu'])[0]))") train_vqvae.py --config config.yaml


*Current Issues:
1. model has no train method
2. incorporate best params
3. Is it possible for our data and hardware to run the study through optuna since it seems 
'''
I know it will split the dataset to num_gpus chunks and each chunk will go to one of the gpus. Is it randomly sampled or sequentially?
I understand that the distributed sampler chunks the dataset for each GPU. However, when using DDP, it loads the entire Dataset on N GPUs N times. Is this how it works?
https://discuss.pytorch.org/t/distributedsampler/90205/3
custom running limit gpu_ids, threshold etc.
'''

- Apply trained VQ-VAE model (**vqvae_reconstruction.py**): To reconstruct images and determine the codebook (embedding space) and indices (latent space) for each image. The trained model is selected from the best model from the previous step
- Train a classifier (**train_classifier.py**): Define the classifier, determine best parameters, and then log all by MLflow, evaluate them on test and select the top one
- Train LLM model, distilbert, (train_distil.py)

Masking Definition:
Random Masking
