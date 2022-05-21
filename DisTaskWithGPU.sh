#Author: Chen Zhitao
#Date: 2021/10/29
#I just want some gpu resource.

function Todo(){
    cd T_SR
    python main.py
}


SleepTime=5
Next='False'
until [ $Next == 'True' ]
do
  GpuUtil=$( nvidia-smi --id=0 -q --display=MEMORY | grep Free | head -n 1 | tr -cd "[0-9]" )
  if [ $GpuUtil -lt 7000 ]
  then
    sleep $SleepTime
  else
    Next='True'
  fi
done

Todo
