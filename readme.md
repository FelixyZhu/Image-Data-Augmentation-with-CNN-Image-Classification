### Data Augmentation for Image Classification Tasks

#### Mini Project for COMP7250 Machine Learning

Author: Felix Zhu

Institution: Hong Kong Baptist University



The project is constructed based on Jupyter Notebook, concentrating on how image data augmentation will influence the training process and result of a CNN. Image dataset using Cifar-10 <[CIFAR-10 and CIFAR-100 datasets (toronto.edu)](https://www.cs.toronto.edu/~kriz/cifar.html)>, deep learning framework using Keras. The outline has been well constructed, feel free to use it for skipping.

Run through main.ipynb to regenerate the results. Download required libraries in one time by `pip install requirements.txt`. Attention: GPU acceleration is suggested as what I did exactly, and corresponding libraries prepared in *requirements.txt*.

GPU acceleration congiuration: (find the libraries' version in *requirements.txt*)

|||
| -------------------------------- | -------------------------------- |
| Python                           | 3.9.19                           |
| GPU                              | RTX3060 Laptop                   |
| NVIDIA-SMI                       | 555.85                           |
| Driver Version                   | 555.85                           |
| CUDA Version (supported version) | 12.5                             |
| CUDA Compilation tools           | 11.2 |
|||



You CAN surely run the codes nicely without GPU (CPU only), but much slower of course. If you wish to do so, please adjust the versions of libraries to meet your need as the libraries ran on my machine (recorded in requirements.txt) may be out of time.

The results involve training process (training loss, training accuracy, validation loss, validation accuracy), loss-epoch plot and accuracy-epoch plot, prediction reports, confusion matrices and prediction visualizations, e.g.:



![image](https://github.com/FelixyZhu/Image-Data-Augmentation-with-CNN-Image-Classification/blob/master/pics/84821aeb-2b5e-4b57-983c-2f13a8cdb637.png?raw=true)

![image](https://github.com/FelixyZhu/Image-Data-Augmentation-with-CNN-Image-Classification/blob/master/pics/b451461e-f2d4-4fd1-884a-7fea825389d9.png?raw=true)

![image](https://github.com/FelixyZhu/Image-Data-Augmentation-with-CNN-Image-Classification/blob/master/pics/fffa0887-6873-47a6-b842-e9a8a6ffca40.png?raw=true)


Stars are welcome and appreciated! Feel free to pull requests for any code suggestions, or contact me by email: felix5196@outlook.com if I am not capable to check the requests.