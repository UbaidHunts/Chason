# This script file is used to reset the fpga state. 
# It is optional but recommended. 
# The previous fpga states can interfere with new dataset output
xbutil reset -d --force
