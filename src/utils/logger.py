
import logging




class Logger:
    def __init__(self, logfile):
        self.logfile = logfile
        
        level = logging.INFO
        # level = logging.DEBUG

        self.logger = logging.getLogger()
        self.logger.setLevel(level)

        formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s: %(message)s", datefmt='%Y/%m/%d %H:%M:%S')
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)


    def get_logger(self):
        self.logger.info(f'Logging to {self.logfile} ...')
        return self.logger

