from urllib import request
import os
url = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz'
def may_download(filename,exp_bytes):
    if not os.path.exists(filename):
        print('begin to download')
        #filename,_ = urllib.request.urlretrieve(url+filename, filename)
        filename, _ = request.urlretrieve(url+filename, filename)
    else:
        print('exsists')
    statinfo = os.stat(filename)
    if statinfo.st_size == exp_bytes:
        print('ok')
    else:
        print('filed')

    return filename

filename = may_download('text8.zip', 31344016)