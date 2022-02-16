from lib.Config.ConfigUtil import ConfigUtil
from lib.Simulators.MeterSimulator import MeterSimulator
from lib.Simulators.StoreSimulator import StoreSimulator
from lib.Storage.StorageUtil import __project_root_folder
from hdfs import InsecureClient

import os
import pathlib
import shutil
from datetime import datetime

MODEL_PATH = 'initModels'
RESULT_PATH = 'upload'
_model_location_template = '{}/' + 'models' + '/{}/{}/{}'

__HDFS_ip = "http://10.10.10.111"
__web_access_port = "50070"

def copyModels():
    streams = ConfigUtil.get_value_by_key('streams')

    save_time = datetime.now()
    date_folder = save_time.strftime("%Y%m%d_%H%M%S.%f")

    # copy meter models
    list_meter_ids = streams.get('meter').get('type_ids')
    meter_pipelines = streams.get('meter').get('pipelines')
    for meter_id in list_meter_ids:
        for pipeline in meter_pipelines:
            pipeline_name = pipeline.get('pipeline_name')

            pathSrc = _model_location_template.format(MODEL_PATH, pipeline_name, meter_id, "")
            if os.path.isdir(pathSrc):
                
                for factor in range(MeterSimulator.METER_REPLICATION_FACTOR):
                    # Meter Factor < Meter Base
                    new_id = MeterSimulator.METER_REPLICATION_FACTOR_BASE*meter_id + factor
                    pathDst = _model_location_template.format(RESULT_PATH, pipeline_name, new_id, date_folder)

                    if os.path.isdir(pathDst) is False:
                        os.makedirs(pathDst) 
                    
                    # file_list = list(pathlib.Path(pathSrc).glob("**/*.*"))
                    file_list = pathlib.Path(pathSrc).glob("**/*.*")
                    for file in file_list:
                        try:
                            shutil.copyfile(str(file), f"{pathDst}/{file.name}")
                            print("copy {} to {}".format(str(file), f"{pathDst}/{file.name}"))
                        except Exception as e:
                            print(f"{file} could not be copied.\n",
                                  f"{type(e).__name__}: {e}")
                    
                    # file = None
    
    # copy store models
    list_store_ids = list(range(1, StoreSimulator.NUMBER_STORES+1))
    store_pipelines = streams.get('store').get('pipelines')
    for store_id in list_store_ids:
        for pipeline in store_pipelines:
            pipeline_name = pipeline.get('pipeline_name')

            pathSrc = _model_location_template.format(MODEL_PATH, pipeline_name, 1, "")
            if os.path.isdir(pathSrc):
                # for id in list_store_ids:
                
                pathDst = _model_location_template.format(RESULT_PATH, pipeline_name, store_id, date_folder)

                if os.path.isdir(pathDst) is False:
                    os.makedirs(pathDst) 
                
                # file_list = list(pathlib.Path(pathSrc).glob("**/*.*"))
                file_list = pathlib.Path(pathSrc).glob("**/*.*")
                for file in file_list:
                    try:
                        shutil.copyfile(str(file), f"{pathDst}/{file.name}")
                        print("copy {} to {}".format(str(file), f"{pathDst}/{file.name}"))
                    except Exception as e:
                        print(f"{file} could not be copied.\n",
                                f"{type(e).__name__}: {e}")
                
                # file = None

def uploadModels():
    client = InsecureClient("{}:{}".format(__HDFS_ip, __web_access_port), user='root')
    # client.walk('/')

    p = pathlib.Path(f"{RESULT_PATH}/models")
    file_list = p.glob('**/*')
    for file in file_list:
        if file.is_dir():
            print(str(file))
            rp_path = file.relative_to(f"{RESULT_PATH}")
            # file.parent
            # file_name = file.name

            # file_path = str(file)

            remote_folder_path = f"{__project_root_folder}/{str(rp_path)}" 
            if client.content(remote_folder_path, strict=False) is None:
                client.upload(remote_folder_path, str(file))
            


    # try:
    #     print(f"{__project_root_folder}")
    #     print(f"{RESULT_PATH}/models")
    #     res = client.upload(f"{__project_root_folder}", f"{RESULT_PATH}/models", overwrite=True)
    #     print(res)
    # except Exception as e:
    #     print(e)

if __name__ == '__main__':
    copyModels()
    uploadModels()

