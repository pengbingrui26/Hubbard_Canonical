# Referenece
[BCM用户手册](https://code.itp.ac.cn/qiyang/install_torque/-/blob/master/user-manual.pdf)

[Slurm中文用户手册](https://docs.slurm.cn/users/)

[modules英文官网](http://modules.sourceforge.net/)

[modules英文手册](https://modules.readthedocs.io/en/latest/)

# ssh 登录
```bash
ssh username@10.105.17.37
```
如果修改本地~/.ssh/config,加入
```bash
Host delta
  HostName 10.105.17.37
  User yangqi
```
可使用
```bash
$ ssh delta
```
直接登录。其他设定和delta服务器没有区别，如果有其他疑问，例如如果需要切换到/bin/zsh作为默认shell，联系运维即可。

---

# module 加载
module就是一些包，这些包目前存放在/cm/shared 和/cm/local文件夹下。我们可以通过命令来查看可用的包
```bash
$ module avail
```
也可以通过module load 加载需要的包
```bash
$ module load julia-1.7.1
```
下面这行命令可以查看目前加载的包。
```bash
$ module list
```
下面是一个例子

```bash
[root@bright90 ~]# julia
bash: julia: command not found...
[root@bright90 ~]# module load julia-1.7.1
[root@bright90 ~]# julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.7.1 (2021-12-22)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

```
由于这些module文件都存储在/cm/shared文件夹下，而这个文件夹又被挂在到了计算节点上，所以本地可以load就意味着计算节点上也生效。

# Slurm 启动交互进程
有的时候我们想调试一段代码，那么可能需要交互式地使用集群上的资源。srun命令可以分配一些内存允许我们登陆到节点之上。比如现在我想用TitanV 1个小时来调试程序：
```bash
srun -p titanv --pty --gres=gpu:1 -t 0-02:00 /bin/bash
```
就可以了


# 作业提交
服务器使用Slurm管理作业调度。与PBS系统类似。
```bash
$ sinfo -Nl # 类似于pbsnodes。查看可用的分区和节点。
$ squeue    # 类似于qstat。查看排队情况
$ sbatch jobfile # 提交jobfile到作业调度系统中，类似于qsub
$ scancel jobid  # 删除jobid对应的job，类似于qdel
$ scontrol show jobid # 查看任务的详细信息 
```

# Slurm jobfile
```bash
[wanglei@bright90 ~]$ sinfo -Nl
Fri Jan 21 07:35:27 2022
NODELIST   NODES PARTITION       STATE CPUS    S:C:T MEMORY TMP_DISK WEIGHT AVAIL_FE REASON              
node001        1    titanv        idle   48   48:1:1 257847   441260      1   (null) none                
node002        1      p100        idle   48   48:1:1 257848   441260      1   (null) none                
node003        1      v100       mixed   80   80:1:1 385611   441260      1   (null) none                
node004        1      v100       mixed   80   80:1:1 385611   441260      1   (null) none                
node005        1      a100        idle  256  256:1:1 515895   898889      1   (null) none  
```
可以看到，目前根据卡的类型，集群有4个分区，分别是titanv，p100，v100和a100。

我们希望在titanv分区上使用4块GPU进行计算，jobfile如下所示：
```bash
#!/bin/bash -l									       	#使用bash执行该脚本
#SBATCH --partition=titanv							#任务提交到titanv分区
#SBATCH --gres=gpu:4		  		        	#使用4块卡
#SBATCH --nodes=1								      	#使用一个节点
#SBATCH --time=1:00:00							  	#总运行时间，单位小时
#SBATCH --job-name=test						  	
echo "The current job ID is $SLURM_JOB_ID"			#从这里开始是执行的脚本
echo "Running on $SLURM_JOB_NUM_NODES nodes:"
echo $SLURM_JOB_NODELIST
echo "Using $SLURM_NTASKS_PER_NODE tasks per node"
nvcc --version
nvidia-smi
echo "A total of $SLURM_NTASKS tasks is used"
echo "CUDA devices $CUDA_VISIBLE_DEVICES"

echo Job started at `date`
python test_pmap.py
echo Job finished at `date`
```
使用
```bash
$ sbatch jobfile
```
提交该脚本，即可看到程序在运行。注：当提交任务到`a100`分区时，还可以进一步标注 `#SBATCH --gres=gpu:A100_40G:4` 或 `#SBATCH --gres=gpu:A100_80G:4`来选择具体使用的卡类型。 


# 关于数据存储的一些说明
每个用户自己的data盘的位置为/data/username 。这个目录在所有机器中同步，数据请写到这里，此硬盘使用raid5备份。需要存档的数据在/archive/username中(raid1)。

# GPU stat:
直接执行slurm_gpustat，可以统计GPU的总体使用情况。
```bash
[yangqi@bright90]$ slurm_gpustat
---------------------------------
Under SLURM management
---------------------------------
There are a total of 39 gpus [up]
7 P100 gpus
8 A100 gpus
8 TitanV gpus
16 V100 gpus
---------------------------------
There are a total of 39 gpus [accessible]
7 P100 gpus
8 A100 gpus
8 TitanV gpus
16 V100 gpus
---------------------------------
Usage by user:
wanglei    [total: 4  (interactive: 0 )] V100: 4
---------------------------------
There are 35 gpus available:
A100: 8 available
P100: 7 available
TitanV: 8 available
V100: 12 available
---------------------------------
```
