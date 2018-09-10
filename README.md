This repository contains an adjusted version of fast-rcnn for our [hospital-people-detector](https://github.com/andresvasquezv/hospital_people_detector/). The original repository is created by Ross Girshick at Microsoft Research, Redmond and can be found [here](https://github.com/rbgirshick/fast-rcnn). 

Please consult the original repo's [README](https://github.com/rbgirshick/fast-rcnn/blob/master/README.md) for information on using fast r-cnn in general. Here, we describe how to get our version running for the hospital-people-detector.

## Installation

# 1. Get Caffe

1.1 Install the necessary requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

1.2 Get the caffe code: Alternative 1:

You can get the original caffe from the BVLC caffe repository. 

  ```Shell
  git clone https://github.com/BVLC/caffe.git
  ```
  
You will need to apply the pull request https://github.com/BVLC/caffe/pull/4163 to make caffe compatible with fast r-cnns special layers: 

  ```Shell
  cd caffe
  git pull origin pull/4163/head
  ```

1.2 Get the caffe code: Alternative 2:

You can also use our caffe fork which includes the above mentioned pull request and is intended to work with fast-rcnn-mobility-aids:

  ```Shell
  git clone https://github.com/marinaKollmitz/caffe.git
  ```

1.3 Compile caffe

  ```Shell
  cd caffe
  cp Makefile.config.example Makefile.config
  ```
Adjust the Makefile.config to your needs. Make sure to uncomment the support for python layers:

  ```Shell
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  ```
Compile caffe and pycaffe

  ```Shell
  make -j8 && make pycaffe
  ```
Consult the [caffe installation instructions](http://caffe.berkeleyvision.org/installation.html#compilation) for more information about compiling caffe.
  
# 2. Get Fast R-CNN

2.1 clone our Fast R-CNN repository

  ```Shell
  git clone https://github.com/marinaKollmitz/fast-rcnn-mobility-aids.git
  ```

2.2 Build the Cython modules

   ```Shell
   cd fast-scnn-mobility-aids/lib
   make
   ```
    
# 3. Make sure the modules can be found

Add caffe and fast-rcnn-mobility-aids to your paths:

  ```Shell
  gedit ~/.bashrc    
  ```

Add the following lines (replace ```<path-to-caffe>``` and ```<path-to-fast-rcnn-mobility-aids>``` with your paths):

  ```Shell
  #Caffe
  export PYTHONPATH=<path-to-caffe>/python:$PYTHONPATH 
  export PATH=<path-to-caffe>/distribute:$PATH 
  #Fast R-CNN
  export PYTHONPATH=<path-to-fast-rcnn-mobility-aids>/lib:$PYTHONPATH
  ```
