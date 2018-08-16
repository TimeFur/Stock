import subprocess

PRE_CMD = 'adb shell "%s"'

'''=======================================================
    Using subprocess as the thread to implement
    -subprocess.PIPE means output type as file
    -shell = True, means the cmd type could be sequence
    it should consider the space & metacharacters
======================================================='''

def shell(cmd):
    
    command = PRE_CMD % (cmd)
    #print command
    
    p = subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    return p.stdout.read()



