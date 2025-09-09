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
        10. Practical Tips:
            - Loss Curves: Plot training and validation loss over epochs to diagnose underfitting or overfitting.
            - Model Checkpointing: Save the model with the lowest validation loss during training.
            - Seed Control: Set random seeds for reproducibility.
            - Normalization: Normalize input data consistently across all splits.
            - Use MLflow or Neptune to track the errors
- Apply trained VQ-VAE model (**vqvae_reconstruction.py**): To reconstruct images and determine the codebook (embedding space) and indices (latent space) for each image. The trained model is selected from the best model from the previous step
- Train a classifier (train_classifier.py): Define the classifier, determine best parameters, and then log all by MLflow, evaluate them on test and select the top one
- Train LLM model, distilbert, (train_distil.py)

Masking Definition:
Random Masking
