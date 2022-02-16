# AppAI 
This repo is used for AppAI system development and research purpose.

## Development Environment
### Resources
* Programming languange - Python 3.6
* Pip3.6
* Python virtual environment
* Ray 
* Docker  
    * Images
        * Redis
        * RabbitMQ
        * Haddop HDFS

### Coding Style
* [PEP 8](https://www.python.org/dev/peps/pep-0008/)


### Setup  
#### Setting up on Ubuntu 18.04 & 16.04
```
    $ sudo add-apt-repository ppa:jonathonf/python-3.6
    $ sudo apt-get update

    ## install python
    $ sudo apt-get install python3.6
    
    # Below steps are optional, follow the steps below if you decide to use VS Code. 
    ## install ubuntu make
    $ sudo apt-get install ubuntu-make 
    
    ## install Visual Studion Code
    $ umake web visual-studio-code
``` 
-------- 

#### Setting up Python virutal environment on Ubuntu 18.04 & 16.04
```
    $ sudo apt-get install python3.6-venv
    $ python3.6 -m venv VIRTUAL-ENV-AppAI
```

#### Using the Python virutal environment 
Activate the python virtual environment
```
    $ source VIRTUAL-ENV-AppAI/bin/activate
```
Once the python virtual environment is activated, the environment name should appears at the beginning of the terminal line in (). 

Upgrade pip to latest
```
    $ pip install -U pip
```

For first time setup, install the project dependency.
```
    $ pip install -r requirements.txt
```

For saving project dependency
```
    $ pip freeze > requirements.txt
```

Deactivate the python virtual environment
```
    $ deactivate
```

Make sure to add both the project and the src/ folders in your PYTHONPATH variable.  
*Note*: This step doesn't set the PYTHONPATH for the whole system. So, you will need to repeat the steps for new terminals.   
If interested, you can add the `export` line to `~/.bashrc` file in the Ubuntu, so it auto-loads when you opens a new terminal.
```
    $ pwd src/
    $ export PYTHONPATH=full/path/to/project/:full/path/to/project/src/
```

-------- 

#### Docker installation: see [link](https://docs.google.com/document/d/1e4feXHSgJi4w3I-PqhcLQazE13dgzuHaY0yX8_sddKU/edit#)

#### Start Resources in Docker
*Method 1*:  Start all the docker container at once: use the script at 10.10.10.111/data/demo-alpha-lite
```
    $ ./run_docker_container.sh
```
*Method 2*: Start them individually, use the commands below. 

1. Start RabbitMQ in Docker
```
    $ docker run -d --hostname rabbit --name docker-rabbit -p 5672:5672 -p 15672:15672 rabbitmq:3-management
```
* For the first time running the command, the machine will pull the rabbitmq image from DockerHub.  
* Default credentials for UI: http://HOSTIP:15672
    * Username: guest
    * Password: guest

-------- 

2. Start Redis in Docker
```
    docker run --name docker-redis -p 6379:6379 -d redis
```
* For the first time running the command, the machine will pull the redis image from DockerHub.  

-------- 

3. Setup Hadoop HDFS  
    3.1 Download our modified standalone Hadoop HDFS image file from 10.10.10.111, path: /data/demo-alpha-lite

    3.2 After downloaded the image, load image into the machine's docker image. 
    ```
        $ docker load < docker-hadoop-no-yarn-img.tar.gz
    ```

    3.3 Start Hadoop HDFS
    ```
        $ docker run --hostname hadoop --name docker-hadoop-no-yarn -p 50010:50010 -p 50020:50020 -p 50070:50070 -p 50075:50075 -p 50090:50090 -d docker-hadoop-no-yarn-img
    ```

    Visit HDFS web UI to verify deployment: http://HOST-IP:50070

    *Important HDFS configuration*:
    * update the `/etc/hosts` file to associate a proper IP to the hostname hadoop

    if using the HDFS on 10.10.10.111, append following line
    ```
        10.10.10.111    hadoop
    ```
    to file `/etc/hosts`.  

    if using local machine, use your own IP instead of `10.10.10.111`

## Configuration Explanation

The configuration has three sections: `resources`, `services`, `streams`.
* To switch between `local` and `database` configuration, developer must change it in the `lib.Config.ConfigUtil.py`
* For Mongo database configuration, the Mongo server connection info is hardcoded in `lib.Common.Auxiliary.py` 


### resources
This is the section for third party services we used in our AppAI system.
1. mysql used for simulation data storage
2. redis used for system internal data caching
3. hdfs used for storing data. 
4. mq used as message broker to exchange messages between AppAI and other outside system/devices

### services
This is the section for the AppAI system services. The AppAI system is designed with MicroService architecture in mind. So, the processing logic is decomposed into services.

1. InputDataHandler is designed to receive data from outside of AppAI system. 
2. StreamProcessor is designed to accept data from InputDataHandler and to perform realtime processing.
3. TrainingActivator is designed to activate training events, by schedule or user-triggered.
    * the specific training schedule configuration is at `stream.XXX.pipelines.[#].training_activation`. 
      The following configuration said to trigger a training event `everyday at 16:38:00 localtime`  
      ```json
       {
             "training_activation":{
                            "day": [],
                            "weekday": [
                                0,1,2,3,4,5,6
                            ],
                            "time": "16:38:00"
            }
       }
        ```
        * `day` is for specifying which day of the month should start the training. Empty for default. Valid inputs: 1 - 31.
        * `weekday` is for specifying which day of the week should start the training. Empty for default. Valid input: 0 - 6. (0 as Monday, 6 as Sunday)
        * `time` is for specifying which time of the day the training should start. It must be in following format "HH:MM:SS"
        `Note:` If `day` is set, whether `weekday` field is set or not, there will be no effect. Between `day` and `weekday`, only one field should be set.   
         
4. ModelTrainer is designed to receive training event requests from TrainingActivator and perform the training.
5. DataRetriever is designed to query realtime processing results and model training related info through Web API.

### streams
This is the section for specifying the incoming data stream and its corresponding pipeline processing logic. 

1. store stream is for sales forecasting scenario
2. meter stream is for abnormal energy consumption detection scenario 

For each pipeline, the training dataset range is configured at `stream.XXX.pipelines.[#].training_data_period`.
The following configuration said to use the `past 1 month` data from the `current activation time` to train the current model.  
```json
{
"training_data_period": {
                        "years": 0,
                        "months": 1,
                        "days": 0,
                        "hours": 0
                    }        
}
```
* years is for the year interval to repeat the training
* months is for the month interval to repeat the training
* days is for the day interval to repeat the training
* hours is for the hour interval to repeat the training
`Note:` default value for each field is 0. If all the fields are set to non-zero, the combination of all fields will be used to calculate the training dataset range
 

## Unit Testing
* Unit testing reference: https://realpython.com/python-testing/#automating-the-execution-of-your-tests
