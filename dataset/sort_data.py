"""
This is a utility file to sort data automaticly. Just place all the files in the 1k_images folder and run this script.
The script will sort the data into host_data and client_data folders. The data will be further sorted into training, validation and test data.
"""

import os

os.makedirs(f'dataset/1k_images/host_data')
os.makedirs(f'dataset/1k_images/client_data')
count = 0

# iterate over all files in 1k_images (if unsorted)
for file in os.listdir('dataset/1k_images'):
    if os.path.isdir(f'dataset/1k_images/{file}'):
        continue
    
    # differentiate between host or client data. 40% of data for host, 60% for clients    
    if count < 320:
        destination = 'host_data/training_data'
    elif 320 <= count < 340:
        destination = 'host_data/valid_data'
    elif 340 <= count < 400:
        destination = 'host_data/test_data'
    elif 400 <= count < 880:
        destination = 'client_data/training_data'
    elif 880 <= count < 940:
        destination = 'client_data/valid_data'
    else:
        destination = 'client_data/test_data'
    
    # check if destination folder exists, if not create necessary folders and files
    if not os.path.exists(f'dataset/1k_images/{destination}'):
        os.makedirs(f'dataset/1k_images/{destination}')
        os.makedirs(f'dataset/1k_images/{destination}/data')
        os.makedirs(f'dataset/1k_images/{destination}/binary_mask')
        os.makedirs(f'dataset/1k_images/{destination}/diameter')
        os.makedirs(f'dataset/1k_images/{destination}/other')
        with open(f'dataset/1k_images/{destination}/binary_mask/annotations_binary_mask.csv', 'w') as f:
            f.write('image_path,binary_mask_path\n')
        with open(f'dataset/1k_images/{destination}/diameter/annotations_diameter.csv', 'w') as f:
            f.write('image_path,diameter\n')
        
    # start moving files to destination folder
    if file.endswith('.png'):
        os.rename(f'dataset/1k_images/{file}', f'dataset/1k_images/{destination}/data/{file}')
        start_index = file.index('_pupil_diameter_')
        end_index = file.index('mm.png')
        diameter_label = file[start_index + len('_pupil_diameter_'):end_index]
        
        with open(f'dataset/1k_images/{destination}/diameter/annotations_diameter.csv', 'a') as f:
            f.write(f'dataset/1k_images/{destination}/data/{file},{diameter_label}\n')
    elif file.endswith('.jpg'):
        os.rename(f'dataset/1k_images/{file}', f'dataset/1k_images/{destination}/binary_mask/{file}')
        start_index = file.index('mm.jpg')
        png_file = file[0:start_index + 3]
        png_file = png_file + 'png'
        
        with open(f'dataset/1k_images/{destination}/binary_mask/annotations_binary_mask.csv', 'a') as f:
            f.write(f'dataset/1k_images/{destination}/data/{png_file},dataset/1k_images/{destination}/binary_mask/{file}\n')
    else:
        os.rename(f'dataset/1k_images/{file}', f'dataset/1k_images/{destination}/other/{file}')
        
    if os.path.exists(f'dataset/1k_images/{destination}/data/{file}') and os.path.exists(f'dataset/1k_images/{destination}/binary_mask/{file}' and os.path.exists(f'dataset/1k_images/{destination}/other/{file}')):
        count += 1