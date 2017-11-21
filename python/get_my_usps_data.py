def usps_data(onehot): 
    import zipfile
    import os
    from PIL import Image
    import PIL.ImageOps  
    import numpy as np
    import tensorflow  as tf
    import matplotlib.pyplot as plt
    
    print("loading USPS DATA ......")
    filename="usps_dataset_handwritten.zip"

    #Defining height,width for resizing the images to 28x28 like MNIST digits
    height=28
    width=28

    #Defining path for extracting dataset zip file
    extract_path = "usps_data"

    #Defining image,label list
    images = []
    img_list = []
    labels = []

    #Extracting given dataset file    
    with zipfile.ZipFile(filename, 'r') as zip:
        zip.extractall(extract_path)

    #Extracting labels,images array needed for training    
    for root, dirs, files in os.walk("."):
        path = root.split(os.sep)

        if "Numerals" in path:
            image_files = [fname for fname in files if fname.find(".png") >= 0]
            for file in image_files:
                labels.append(int(path[-1]))
                images.append(os.path.join(*path, file)) 

    #Resizing images like MNIST dataset   
    for idx, imgs in enumerate(images):
        img = Image.open(imgs).convert('L') 
        img = img.resize((height, width), Image.ANTIALIAS)
        img_data = list(img.getdata())
        img_list.append(img_data)

    #Storing image and labels in arrays to be used for training   
    USPS_img_array = np.array(img_list)
    USPS_img_array = np.subtract(255, USPS_img_array)
    USPS_label_array = np.array(labels)

    #Give Output as onehot vector
    if(onehot):
        nb_classes = 10
        targets = np.array(USPS_label_array).reshape(-1)
        aa = np.eye(nb_classes)[targets]
        USPS_label_array = np.array(aa, dtype=np.int32)

    #Normalize USPS data to fit range 0-1
    USPS_img_array = np.float_(np.array(USPS_img_array))
    for z in range(len(USPS_img_array)):
        USPS_img_array[z] /= 255.0 
    
    print("USPS DATA Loaded")    
    return USPS_img_array, USPS_label_array
