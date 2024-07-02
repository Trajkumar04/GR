Welcome to HandSignia!

HandSignia is a program that allows users to better express themselves in video's using hand gestures. Users can
perform 1 of 8 gestures, and have visual effects show up on their video feed.

There are currently 8 gestures that are available.

Thumbs Up
Thumbs Down
Heart Hand
Peace/Victory
Rock and Roll
Ok
Stop
Shaka (Similar to the call me hand sign)

Bugs:

It should be noted that the application has a little bit of difficulty with the peace, rock, and shaka hand signs.
Often the model will play the sound effect associated with the ok hand gesture. 
Furthermore, the application is unable to detect the thumbs up hand gesture, even though it has been trained to do so. The code for the effect
is there, its just that the model doesn't detect it. This is something I hope to fix in later versions of the code.
The model may take a bit to detect the gesture you perform. Please be patient.


Instructions:

In order to run this application, you must activate the virtual environment. This can be done with the command below.
.\.venv\Scripts\activate 

Afterwards, simply run the video_capture.py file located in the src folder.
You can either cd into the src folder and run the code with line python video_capture.py in terminal, or use an IDE like visual studio code
or pycharm.

To close the application, simply press q on the keyboard.

Additional Requirements:

Check the requirement.txt file and confirm that you have these specific versions downloaded.

Website:

I have made a website that goes into more detail about the project, including talking about how the project
was developed, information about the creator, and a demo video. The link is listed below.

https://sites.google.com/view/handsignia/home
