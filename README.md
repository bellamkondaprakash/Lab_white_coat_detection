## Lab white coat detection

 > Basically it for detection of Lab white coat and lab white apron coat.
 
 > End_2_End development of start scrap untill predictions and containerzation using docker or kubl. 

 
 * Step 1: run the [Download_libaries.sh](https://github.com/bellamkondaprakash/Lab_white_coat_detection/blob/master/Download_libaries.sh) file to not struck with error while running the code
 ```bash
 source env/bin/activate
 sh Download_libaries.sh
 ```
 * Step 2: scrap the images from the google with query img and number of images to scrap and directory to save[Image scraper](https://github.com/bellamkondaprakash/Lab_white_coat_detection/blob/master/google_image_scrap.py)
 
 ```bash
 source env/bin/activate
 python google_image_scrap.py -s='lab apron' -n=10 -d='directory_to_store_images'
 
 ```
 
 * Step 3: Images augmentation with scraped the images and number of times[Augmentation of images](https://github.com/bellamkondaprakash/Lab_white_coat_detection/blob/master/img_augment.py) 
 
 ```bash
 source env/bin/activate
 python img_augment.py -i='load direcory of the scraped images' -n='number of times images to augment of diff verticles' -d='directory_to_store_images'
 ```
 
 * Step 4: Run the Image vgg19.py file 
 ```bash
 source env/bin/activate
 python vgg19.py 
 ```
 ---
 ```
 ToDo's:
    - Model Improvement
    - Conternization
 ```
