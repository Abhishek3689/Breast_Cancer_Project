import os,sys
import logging
from datetime import datetime

log_name=f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
log_path=os.path.join('logs',log_name)
os.makedirs(os.path.dirname(log_path),exist_ok=True)

log_file_path=os.path.join(log_path,log_name)

logging.basicConfig(filename=log_file_path,level=logging.INFO,format="[%(asctime)s ] %(lineno)d %(name)s -%(levelname)s -%(message)s")