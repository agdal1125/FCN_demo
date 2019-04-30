# Fully Convolutional Neural Network for Semantic Segmentation

#### modified codes from https://github.com/warmspringwinds/tf-image-segmentation

### File Structure
    .
    ├── data                # your testing data dir
    |   └──                 # your image data goes here 
    |
    ├── fcn_16s_checkpoint     # place pretrained fcn16 model here :
    |     └── model_fcn16s_final.ckpt.data... # download 
    |     └── model_fcn16s_final.ckpt.index   # https://www.dropbox.com/s/tmhblqcwqvt2zjo/fcn_16s.tar.gz?dl=0 and 
    |     └── model_fcn16s_final.ckpt.meta    # unzip it here
    |     
    ├── models              
    |     └── slim
    |     └──  ...
    |     └──  ...
    |     
    ├── test.py                # python script for testing trained model
    |
    |
    ├── fcn8_train.py
    |
    ├── tf_image_segmentation
    |      └── generated       # generated image files
    |      └── saver           # save folder
    |      └── log_folder_fcn8 # log folder
    |      └── background_extracted # generated background images without person
    |      └── person_extracted # generated person images without background
    |     
    └── README.md
  

### Dependencies
- tensorflow r.0.12 or later version
- $pip3 install scikit-image Pillow matplotlib, numpy, tensorflow


### Path Setting

Modify PATHS in "fcn8_train.py"

- VGG16 checkpoint PATH 생성 (http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)

models/slim 패스 설정

## Usage
```sh
$ conda create -n YOUR_ENV_NAME
$ source activate YOUR_ENV_NAME
$ cd YOUR_WORKING_DIR/FCN_demo/

# To Test segmentation of your image data, place the test data in  "~/FCN_demo/data/" directory 

$ python3 test.py --dir YOUR_DIRECTORY_TO_FCN_DEMO

ex) $ python3 test.py --dir /home/nowgeun1/Desktop/FCN_demo

# You can find the prediction of your image data in  "~/FCN_demo/tf_image_segmentation/generated/"
# You can find extracted background image in "~/FCN_demo/tf_image_segmentation/background_extracted/"
# You can find person background image in "~/FCN_demo/tf_image_segmentation/person_extracted/"
```

### Server Usage (*For Crevasse Only*)

1.개발환경접속
$source activate fcn

2.모듈 파일 위치로 이동
$cd /home/nowgeun1/Desktop/FCN/

3.pretrained 된 모델 테스트
$python3 test.py

4.예측 확인
/home/nowgeun1/Desktop/FCN/tf_image_segmentation/generated/
/home/nowgeun1/Desktop/FCN/tf_image_segmentation/background_extracted/
/home/nowgeun1/Desktop/FCN/tf_image_segmentation/person_extracted/

**User Input Data 활용법

테스트 이미지 파일 저장 폴더:

YOUR_WORKING_DIR/FCN_demo/data/

ex) /home/nowgeun1/Desktop/FCN_demo/data/


예측 결과 폴더:

YOUR_WORKING_DIR/FCN_demo/tf_image_segmentation/generated/

ex) /home/nowgeun1/Desktop/FCN_demo/tf_image_segmentation/generated/



References:

http://warmspringwinds.github.io/tensorflow/tf-slim/2017/01/23/fully-convolutional-networks-(fcns)-for-image-segmentation/
https://github.com/warmspringwinds/tf-image-segmentation
