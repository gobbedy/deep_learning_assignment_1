#!/bin/bash

set -o pipefail
me=$(basename ${0%%@@*})
full_me=${0%%@@*}
me_dir=$(dirname $(readlink -f ${0%%@@*}))

# create output directory tree
datetime_suffix=$(date +%b%d_%H%M%S)
output_dir=${me_dir}/regress/${datetime_suffix}
mkdir ${output_dir}
mkdir ${output_dir}/dispy # for all dispy logs
mkdir ${output_dir}/autogen # for any autogenerated data

# simulation parameters
#nodes=16
#cpus=768
#time="00:06:00" # hours:minutes:seconds
#nodes=1
cpus=1
time="00:05:00" # hours:minutes:seconds (3hrs not completed for 100K)
mem=1gb # MaxRSS 30GB for 100K (multi), 1.9GB for 100K (multi/dispy), 190MB for 10k (multi/dispy), 3.6GB for 10k (multi)
email=yes
test_mode=no
job_name="assignment1_simulation"
sbatch_script="simulation.sbatch"
python_options="" # eg -h|--short, -s|--sanity, -p|--profile
logfile=${output_dir}/${job_name}.log

# print latest logfile to file to find it easily
#echo $logfile > latest_log.txt


# launch job
export="output_dir=\"${output_dir}\",python_options=\""${python_options}"\""
mail=''
if [[ ${email} == yes ]]; then
  mail="--mail"
fi
test=''
if [[ ${test_mode} == yes ]]; then
  test="--test"
fi

#echo "$me: launching the following command:"
slurm.sh --cmd sbatch -t "${time}" -j "${job_name}" -o "${logfile}" -n "${nodes}" -c "${cpus}" -m "${mem}" -e "${export}" ${mail} ${test} "${sbatch_script}" |& tee -a ${logfile}

echo "" >> ${logfile}
echo "${me}: JOB OUTPUT:" >> ${logfile}