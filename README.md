# professor-frog

Welcome to Professor Frog / Grade my Frog / Rate My Frog, Professor / Frog Professor !

This is a Standard Frog Grading System, allowing users to finally receive feedback on their beautiful frog drawings.

I created four keras ML models using around 200 frog images I downloaded and organized to evaluate performance by various artistic metrics.

I then created a web interface which allows users to submit a drawing through a post request, which would be taken by a python flask application which saved the image in the /uploads folder, ran the machine learning models on it, and sent back an array containing the professor evaluation and numerical values of grades.

I used a jquery ajax method to handle the /predict API call and handle the results, using plotly to display them. 

Because I had to find, download and grade the training images myself, I was very limited in dataset size. I hope to continue working on this, incorporating more data and models.

To use this program, navigate to the professor-frog-neural-network directory, run python app.py in the terminal, and copy-paste the local host url from the python output into your web browser.
