import subprocess
import os
import commands

PRE_CMD = 'adb shell "%s"'

'''=======================================================
    Using subprocess as the thread to implement
    -subprocess.PIPE means output type as file
    -shell = True, means the cmd type could be sequence
    it should consider the space & metacharacters
    <This method would read until the msg done>
======================================================='''

def shell(cmd):
    
    command = PRE_CMD % (cmd)
    print command
    
    p = subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    
    return p.stdout.read()

def keep_listen_shell(cmd):
    p = subprocess.Popen(cmd, shell = True, stdout = subprocess.PIPE)
    while True:
        l = p.stdout.readline()
        if not l:
            break
        print l

try:
    #keep_listen_shell("adb shell getevent -r")
    keep_listen_shell("adb devices")
except:
    pass
