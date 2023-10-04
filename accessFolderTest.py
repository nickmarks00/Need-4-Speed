# Python program to explain os.listdir() method 
    
# importing os module 
import os
  
# Get the list of all files and directories
# in the root directory
path = "/home/david/Desktop"
dir_list = os.listdir(path)
  
print("Files and directories in '", path, "' :") 
  
# print the list
print(len(dir_list))