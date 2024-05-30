To train a network with supervised learning for the purpose of imitation learning, please run
"Autokeras_IL_Training2"

The code is commented so please refer to the comments in the code. The general flow of the model is

1.Data loading
2.Data pre processing (normalisation of images and labels)
3.Training and evaluation.
4.Saving

some things to note:
1.Please install all the dependencies stated at the start of the notebook, they may be incomplete
so please install the needed extra dependencies when you do encounter an error.

2.Put the dataset folder in the samedirectory of the folder, please name to your preference.

3.please make note where the folder of the images are, and the corresponding csv files.

4.copy the path of both things stated in 4 
(make sure they are complete path "C:/...." and not just "/folder" so there wont be issues 
with accessing the directory)

5.Add your dataset by specifying the image folder path and the corresponding csv file as
per how it's done in the notebook

6.the notebook should run just fine. 

7.Autokeras will create a folder called "image regressor" which holds all the training parameters
during fit. by default, the notebook sets the "overwrite" to true, so it will overwrite
any other image regressor files that have the same name. IN THE CASE it stops in the middle of training. you can
resume training by loading the image regressor file (i.e setting the overwrite to false)
and run "fit" again. It shuold continue from when the training was interrupted.
*note it may say 0 epochs, but the training loss and accuracy will start from the last
cycle before the interrupt

7.to view the training and validation loss, please refer to the comments in the notebook. 

8.for fine tuning, please refer to the notebook's comments.