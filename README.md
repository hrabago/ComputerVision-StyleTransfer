# ComputerVision-StyleTransfer
Columbia University COMSW4731 Final Project

# Project Statement
For our final project we will be implementing the suggested project topic of Artistic Style Transfer. We will extract the style of famous paintings and apply that style to the content of photographs taken near Morningside Campus. We want 20 images as output for evaluation purposes, therefore we will apply 5 different painting styles to each of the contents of the 4 original photos.

# Methods
We will be basing our project on the VGG network as used by the paper A Neural Algorithm of Artistic Style. The VGG network is a deep neural network with 16 to 19 layers. We will use the 16 convolutional layers and 5 average pooling layers. The VGG network comes from the paper Very Deep Convolutional Network for Large-Scale Image Recognition.[1] 

Keras will be used to speed up implementation of the VGG network. By using Keras we will be able to work faster and simplify building a 16 to 19 layers deep neural network. Which would require more extensive time or personnel otherwise.
A Neural Algorithm of Artistic Style [2] will be our main reference work for this project.
Preserving Color in Neural Artistic Style Transfer [3] will be used to extend the work towards preserving color, given that the first stage of the project works as expected.

# Evaluations
We will test the results by asking at least 5 of our peers to give two types of score to each of the 20 generated images. The first score will be based on the transfer of the artistic style and the second score will be based on the preservation of the content. The score for transfer of style will range from 1 to 10, with 10 being “All Style Transferred”, 5 being “Style Somewhat Transferred”, and 0 being “Style Not Transferred”. The score for preservation of content will range from 1 to 10, with 10 being “All Content Preserved”, 5 being “Content Somewhat Preserved”, and 0 being “Content Not Preserved”. After gathering the results, we expect to achieve at least 70% for both Transfer of Style and Content Preservation.

# Group Members and division of tasks
This group is formed by Eduardo Despradel and Hector Rabago. Both of the team members are responsible for researching related work. We will work in face-to-face mini-hackathon like sessions for the coding aspect to advance on the project. We will document our progress in a google docs, explaining what we learn, and the steps we have to take. Eduardo will document the progress the project is taking. The results will be gathered and analyzed by Hector.

