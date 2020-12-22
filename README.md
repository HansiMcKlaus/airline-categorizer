# Airline Categorizer

Categorizes Airplanes to Airline Companies via their empennage by calculating the mean color (value) of the empennage and comparing them to a few different airline companies (Hong Kong Airlines, Lufthansa, Thai Airways, American Airlines, China Southern Airlines, United Airlines, easyJet).

This program is just an extension I made for fun from an exercise I did at Uni. It is not really user-friendly, nor in any way complete (though it is rather easily expandable, I guess) and the easiest way to use it would be in Spyder directly.



### How to use

First you need an binary image (`.png`) of an empennage and name it `leitwerkMaske.png`. The image should be roughly the same size and rotation as the empennages on the planes. The program itself will create masks from 80-120% size of the image in both orientation so it does not need to be totally accurate.

Create a folder in the same directory as the code with all the images that need to be categorized and paste the name of the directory into line 30 to replace `Ordner` with the name of the directory. The images need to be in `.jpg`-format.

Execute the code. After a while (~3 seconds/image), all images will be categorized. To see which images were categorized to what Airline, you can print/inspect the array `vorhersage`.



### Dependencies

The program uses the following libraries which may not come automatically with python:

- numpy
- glob
- skimage
- matplotlib
- scipy
- time