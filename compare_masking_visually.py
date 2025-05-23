import matplotlib
matplotlib.use('Agg')
from PIL import Image
import os
import re
from matplotlib import pyplot as plt

# Define the paths to your images
image_folder = 'image/recons/comparison'

# Define the main images
main_images = [str(i).zfill(5) for i in [462,1671,1836,4970,5852,7777,8513,8685,9469,9644]]

# Define the masking approaches and percentages
masking_approaches = ['Additive', 'Selective', 'Random']
masking_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100]

# Function to create combined image for a main image
def create_combined_image(main_image):
    # Initialize a list to hold the images
    fig, axes = plt.subplots(6, 12, figsize=(24, 12))
    fig.suptitle(f'Combined Image for {main_image}', fontsize=16)
    
    for approach_idx, approach in enumerate(masking_approaches):
        for percentage_idx, percentage in enumerate(masking_percentages):
            pattern_random = re.compile(f'{main_image}_MaskPercentage={percentage}_{approach}Masking_label=(.*).png')
            pattern_transformer = re.compile(f'{main_image}_MaskPercentage={percentage}_{approach}MaskingWithTransformer_label=(.*).png')
            
            # Find the file that matches the pattern
            for file_name in os.listdir(image_folder):
                match_random = pattern_random.match(file_name)
                match_transformer = pattern_transformer.match(file_name)
                
                if match_random:
                    file_path = os.path.join(image_folder, file_name)
                    image = Image.open(file_path)
                    axes[approach_idx * 2, percentage_idx].imshow(image)
                    axes[approach_idx * 2, percentage_idx].axis('off')
                    label_value = match_random.group(1)
                    if approach == 'Additive':
                        fig.text(percentage_idx / 12 + 0.04, 0.75, f'Label: {label_value}', ha='center', fontsize=9)
                    elif approach == 'Selective':
                        fig.text(percentage_idx / 12 + 0.04, 0.46, f'Label: {label_value}', ha='center', fontsize=9)
                    elif approach == 'Random':
                        fig.text(percentage_idx / 12 + 0.04, 0.17, f'Label: {label_value}', ha='center', fontsize=9)
                    if approach_idx == 0:
                        axes[approach_idx * 2, percentage_idx].set_title(f'{percentage}% Masking')
                
                elif match_transformer:
                    file_path = os.path.join(image_folder, file_name)
                    image = Image.open(file_path)
                    axes[approach_idx * 2 + 1, percentage_idx].imshow(image)
                    axes[approach_idx * 2 + 1, percentage_idx].axis('off')
                    label_value = match_transformer.group(1)
                    if approach == 'Additive':
                        fig.text(percentage_idx / 12 + 0.04, 0.61, f'Label: {label_value}', ha='center', fontsize=9)
                    elif approach == 'Selective':
                        fig.text(percentage_idx / 12 + 0.04, 0.32, f'Label: {label_value}', ha='center', fontsize=9)
                    elif approach == 'Random':
                        fig.text(percentage_idx / 12 + 0.04, 0.03, f'Label: {label_value}', ha='center', fontsize=9)

    # Add ylabels outside the axes grid
    fig.text(0, 0.82, 'Additive Masking', va='center', rotation='vertical', fontsize=8)
    fig.text(0, 0.68, 'Additive Masking + Transformer', va='center', rotation='vertical', fontsize=8)
    fig.text(0, 0.53, 'Selective Masking', va='center', rotation='vertical', fontsize=8)
    fig.text(0, 0.39, 'Selective Masking + Transformer', va='center', rotation='vertical', fontsize=8)
    fig.text(0, 0.24, 'Random Masking', va='center', rotation='vertical', fontsize=8)
    fig.text(0, 0.09, 'Random Masking + Transformer', va='center', rotation='vertical', fontsize=8)

    # Adjust layout to ensure labels are not clipped
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the combined image
    # output_path = os.path.join(image_folder, f'combined_{main_image}.png')
    plt.savefig(f'combined_{main_image}.png', bbox_inches='tight')
    print(f"Combined image for {main_image} created successfully!")
    plt.close()

# Run the function for each main image
for main_image in main_images:
    create_combined_image(main_image)
