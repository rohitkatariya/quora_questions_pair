import datetime
import time
def getDateString(date_to_convert=None,dtformat="%d_%m_%Y" ):
    if date_to_convert:
        today = date_to_convert
    else:
        today = datetime.date.today()
    return str(today.strftime(dtformat))

def getDateFromDateString(date_str, dtformat="%d_%m_%Y"):
    try:
        my_date = datetime.datetime.strptime(date_str, dtformat)
    except:
        return None
    return my_date
def getTimeInSec():
	return time.time()
