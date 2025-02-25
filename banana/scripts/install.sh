#! /bin/bash

echo
echo THIS SCRIPT IS AN ATTEMPT TO AUTOMATIZE THE 
echo INSTALLATION OF TENSORPHY / HAMLET
echo 
echo NOT EVERYTHING MIGHT WORK, YOU MAY HOWEVER 
echo USE IT A RECEPY 
echo 
echo IT IS DESIGNED TO WORK ON THE NEWTON CLUSTERS
echo OF THE AIP
echo 
echo TENSORPHY / HAMLET SHOULD NOT BE \(TOO\) DEPENDANT 
echo ON THE EXACT VERSION OF THE PACKAGES
echo 
echo THIS IS HOWEVER NOT THE CASE FOR TENSORFLOW / TENSORFLOW-PROBABILITY
echo WHICH NEED TO BE COMPATIBLE WITH THE CUDA LIBRARIES
echo CHECK THEIR WEBSITES TO KNOW MORE 
echo
echo HAMLET/ WILL BE INSTALLED NEXT TO TENSOPHY/
echo
echo WRITTEN IN Sep. 2023
echo

# conda / python environment
conda_path=/opt/aconda3/etc/profile.d/conda.sh
env_name=py38-tf
packages_to_install="numpy scipy matplotlib camb astropy dill ipython json5 tensorflow==2.12 tensorflow-probability==0.20"
cpip=$HOME/.conda/envs/$env_name/bin/pip 
#
# hamlet source
hamlet_source='https://github.com/anvvalade/hamlet.git'

# modifying environment paths
# maybe not necessary
# user=$(id -u)
# sed -i "s/avalade/$user" $env_yml_path
# sed -i "s/py38-tf/$env_name" $env_yml_path

# setting up the environment
source $conda_path
yes | conda create -n $env_name python=3.8
conda activate $env_name
echo THE NEXT OPERATION MAY TAKE QUITE SOME TIME... 
yes | conda install -c conda-forge cudatoolkit=11.2 cudnn=8.8
cpip install $packages_to_install

# installing tensorphy
cd ../../ 
echo "# automatically set path from install script:" >> tensorphy/common.py
echo "tensorphy_path = \"$PWD/tensorphy/\"" >> tensorphy/common.py
python setup.py install --user

# installing hamlet
cd ../
git clone $hamlet_source

# Done!
echo
echo Hamlet, Tensorphy, etc are only available '*within*' the conda environment
echo Enable it with: source $conda_path && conda activate $env_name
echo Remark: rsync does not work properly within this environment
echo
echo You may want to add to your .bashrc the alias:
echo alias py='bash --rcfile <(echo '\''. ~/.bashrc; source /opt/aconda3/etc/profile.d/conda.sh; conda activate py38-tf'\'')'
echo which allows you to enter a properly initialized bash env, you can exit with ctrl+D or \"exit\"
echo Please copy it from this script to have to the correct quotes, and change the conda source and the environment name if necessary
echo
echo Once you\'re in the correct environment, check the installation with: 'ipython -c "import tensorphy"'
echo 
echo To submit the job to slurm, use \(or modify\) the tensorphy/scripts/submit_python_job.sh 
echo 
echo Good luck!
