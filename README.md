# ML-Basic
Implementing Neural Networks in TensorFlow

To create a Docker setup to run machine learning applications with the technologies you mentioned (numpy, scipy, matplotlib, ipython, scikit-learn, pandas), here's an example of a Dockerfile and a docker-compose.yml file for running it locally:
________________________________________
Steps to Use
1.	Prepare the Files
o	Place the Dockerfile, docker-compose.yml, and requirements.txt in the same directory.
o	(Optional) Add your Python scripts or Jupyter Notebooks to the directory.
2.	Build the Docker Image Run the following command to build the image:
bash
Copy code
docker-compose build
3.	Run the Application Start the container and access the environment:
bash
Copy code
docker-compose up
4.	Access Jupyter Notebook
o	Open your browser and go to http://localhost:8888.
o	Use the token provided in the terminal logs to log in.
5.	Stop the Container When you're done, stop the container with:
bash
Copy code
docker-compose down
________________________________________
Using the Environment
•	For Scripts: Place your Python scripts in the project directory and run them inside the container:
bash
Copy code
docker-compose exec ml-app python your_script.py
•	For Jupyter Notebooks: You can create or upload .ipynb files directly through the Jupyter interface.

Test:
Test Jupyter Notebook
If your CMD in the Dockerfile is set to start Jupyter Notebook, you can test it as follows:
Start the Container
Run:
bash
Copy code
docker-compose up
Access Jupyter
•	Open your web browser and go to http://localhost:8888.
•	Look in the logs printed in the terminal for a URL that contains a token (e.g., http://127.0.0.1:8888/?token=...). Use this link to access Jupyter Notebook.
•	Once inside Jupyter, create a new notebook and run:
python
Copy code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print("All packages are installed and working!")
If it prints the message without errors, the setup works.
________________________________________
2. Test with a Python Script
Create a Python Test Script
Create a file named test.py in the same directory as your docker-compose.yml:
python
Copy code
# test.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Simple test for sklearn
X = np.array([[1], [2], [3], [4]])
y = np.array([1.5, 3.5, 5.5, 7.5])

model = LinearRegression()
model.fit(X, y)

print("Model trained successfully!")
print(f"Model coefficients: {model.coef_}, Intercept: {model.intercept_}")

# Simple pandas test
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
print("Pandas DataFrame:\n", df)

# Simple matplotlib test
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("Matplotlib Test")
plt.savefig("test_plot.png")
print("Matplotlib test plot saved as 'test_plot.png'")

![test_plot](https://github.com/user-attachments/assets/47b8d0e5-daa1-4eb4-8491-9c61106ba233)
________________________________________
Run the Script
Run the container and execute the script inside it:
bash
Copy code
docker-compose up -d
docker-compose exec ml-app python test.py
You should see outputs confirming the model training, a DataFrame printed to the terminal, and a plot saved as test_plot.png in the project directory.
________________________________________
3. Check Installed Packages
To verify all Python packages are installed in the container, log into the container and check the installed modules:
bash
Copy code
docker-compose exec ml-app bash
pip list
This will show all installed packages and their versions.
________________________________________
4. Debugging
•	If any library is missing, double-check your requirements.txt file.
•	Use docker-compose logs to view any issues during container startup.
Let me know if you run into any errors or need further help!


