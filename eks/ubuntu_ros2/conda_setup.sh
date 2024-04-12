apt-get install -y gpustat

__conda_setup="$('/opt/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/opt/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/opt/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda init bash
conda create -n eureka python=3.8 -y
conda activate eureka
cp /opt/anaconda3/envs/eureka/lib/libpython3.8.so.1.0 /usr/lib

cd /mnt/fsx/addf-dcv-demo-us-east-2/isaacgym/python
pip install -e .

cd /mnt/fsx/addf-dcv-demo-us-east-2/Eureka
pip install -e .
cd isaacgymenvs; pip install -e .
cd ../rl_games; pip install -e .
pip install xvfbwrapper ipython progressbar2

cp -r /mnt/fsx/addf-dcv-demo-us-east-2/fbx /opt/anaconda3/envs/eureka/lib/python3.8/site-packages/
cp /mnt/fsx/addf-dcv-demo-us-east-2/FbxCommon.py /opt/anaconda3/envs/eureka/lib/python3.8/site-packages/

mkdir /opt/health-check
touch /opt/health-check/ready
echo "Setup done..."

set +x
while true
do
  sleep 5
done
