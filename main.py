import time
import logging

from HW1.main import main

if __name__ == '__main__':
    start_time = time.time()
    logging.basicConfig(format='%(asctime)s:: %(message)s', datefmt='%m/%d/%Y %I:%M:%S',
                        filename="logger_{}.log".format(start_time), filemode='w', level=logging.DEBUG)
    main()
    logging.info("--- %s seconds ---" % (time.time() - start_time))
    print("--- %s seconds ---" % (time.time() - start_time))
