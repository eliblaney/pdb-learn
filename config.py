import logging
import os
import datetime
from alive_progress import config_handler

LOG_DIR = 'logs/'

now = datetime.datetime.now()

if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

logging.basicConfig(filename="{}/{}.log".format(LOG_DIR, now), level=logging.DEBUG, filemode='w', format='[%(asctime)s %(levelname)8s] %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s] %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

config_handler.set_global(unknown='balls_scrolling', bar='blocks')

logging.debug("Program started %s", now)

