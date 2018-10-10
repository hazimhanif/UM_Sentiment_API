# UM Sentiment Analysis Web API Deployment

This is the Application Programming Interface (API) development for a sentiment analysis framework managed by Security Research Group (SECREG), Faculty of Computer Science and Information Technology, University of Malaya, Malaysia.  
The sentiment analysis API implements a Word2Seq Convolution Neural Network trained with cross-domain datasets (~1 million instances) using GPU-based Tensorflow.  
Contributors:  
* Hazim Hanif
* Nor Badrul Anuar

## Installation


### Installing prerequisite softwares
* Python 3.6 (This deployment uses Anaconda. Get the distribution here https://www.anaconda.com/download/)
* MySQL (Latest version)

### Create new MySQL user
Create a new user to access the database. Avoid using root user to access the database.  
```
mysql
CREATE USER 'username'@'%' identified by 'yourpassword';
GRANT ALL PRIVILEGES ON *.* to 'username'@'%';
FLUSH PRIVILEGES;
```

### Create new database in MySQL to store data
Customize the `database` name according to your own.  
```
mysql
CREATE DATABASE sentiment;
USE sentiment;
CREATE TABLE `API_text` (
	`ID` INT(11) NOT NULL AUTO_INCREMENT,
	`Text` VARCHAR(1000) NULL DEFAULT NULL,
	`Text_hash` CHAR(40) NULL DEFAULT NULL,
	`Pred_sentiment` CHAR(50) NULL DEFAULT NULL,
	`Prob_positive` FLOAT NULL DEFAULT NULL,
	`Prob_negative` FLOAT UNSIGNED NULL DEFAULT NULL,
	`IP_address` CHAR(50) NULL DEFAULT NULL,
	`Timestamp` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	INDEX `Index 1` (`ID`)
)
COMMIT;
```

### Creating a new virtual environment
This step is important to isolate your deployment environment from root/base environment.  
The command to create is: `conda create -n <YOUR ENVIRONMENT NAME> python=3.6`  
Activate the environment upon completing the installation.  
```
conda create -n myEnv python=3.6
conda activate myEnv
```

You can deactivate the environment layer using: `conda deactivate`.  

### Installing prerequisite python libraries
* Tensorflow 1.10
* Keras 2.2.2
* Numpy
* Flask
* PyMySQL

Feel free to use `pip`, however this deployment uses `conda` as the main package manager.  
Try to stick with `conda` to avoid any dependency issues.  
```
conda install tensorflow=1.10 keras=2.2.2 numpy flask pymysql
```

### Cloning the repo
```
git clone https://github.com/hazimhanif/UM_Sentiment_API.git
```

### Configuring your database settings for the API
Use your favourite text editor for this task.  
Edit the `app.py` file in the `UM_Sentiment_API/Scripts` directory.  
Change the values according to your database.
```
db_host = 'yourHost'
db_username = 'userName'
db_pass = 'userPass'
db_name = 'dbName'
```

### Configuring your Flask connection settings for the API
Edit the `app.py` file in the `UM_Sentiment_API/Scripts` directory.  
Edit the `host` value to where do you want to host your API.  

* *localhost* = Local IPv4  
* *127.0.0.1* = Local IPv4  
* *0.0.0.0* = Public IPv4  
* *[::]* = Public IPv4/v6 (Dual-stack deployment)  

As for `port`, assign any unused port to bind. Default is `5000`.   
```
app.run(host='localhost', port=5000)
```

### Running the API
It's easy.  
Just run the python script (`api.py`) available in the `UM_Sentiment_API/Scripts` directory.
```
python api.py
```

### Using the API
Simplest example to use the API is through the browser.  
Define an API call URL.  
e.g: `http://localhost:5000/api/v1/sentiment?text=<InsertYourTextHere>` 

You need to use `%20` as *SPACE* in the text.  
e.g: `http://localhost:5000/api/v1/sentiment?text=The%20movie%20is%20very%20cool`  

# Thank You!
