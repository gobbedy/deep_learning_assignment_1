#!/bin/bash

set -o pipefail
me=$(basename ${0%%@@*})
full_me=${0%%@@*}
me_dir=$(dirname $(readlink -f ${0%%@@*}))

######################################################################
# Convenience functions
######################################################################
### exit elegantly
function die {
  err_msg="$@"
  printf "$me: %b\n" "${err_msg}" >&2
  exit 1
}

# Usage: info "string message"
function info
{
  printf "${me}: INFO - %s\n" "$@"
}

######################################################################
# Get node prefix, node suffix and login node
######################################################################

# get node prefix (eg gra, cdr, nia)
node_prefix=$(echo ${SLURM_JOB_NODELIST} | cut -c1-3)

# get node suffix for python, and login node
# eg in python cedar nodes are "cdr1132.int.cedar.computecanada.ca" instead of "cdr1132"
case "$node_prefix" in
  gra)
    python_node_suffix=''
    login_node="FILL THIS IN LATER"
  ;;
  cdr)
    python_node_suffix='.int.cedar.computecanada.ca'
    login_node="cedar1"
  ;;
esac

######################################################################
# Setup cleanup routine to trap on any exit condition (even normal)
######################################################################

function cleanupOnExit
{

  echo "ENTERING DA TRAP" > /home/gobbedy/projects/def-yymao/gobbedy/thesis-scratch/portfolio/tmpppp

  error=0
  local signal="$1"
  if [[ "${signal}" != "EXIT" ]]; then
    error=1
    printf "\nERROR in ${me}: User interrupted (${signal})\n" >&2
  fi

  # kill tmux sessions
  if [[ -n ${launched_dispynodes} ]]; then
    for node_name in ${node_name_list[@]} ; do
      ssh ${node_name} "tmux kill-session -t dispynode_session"
    done
  fi

  # summary results for end user
  if [[ -n ${mail} ]]; then
    # send e-mail to end user: since compute node doesn't have internet, must ssh to login node
    ssh ${login_node} "summarize_regression.sh ${error} ${SLURM_JOB_NAME} ${SLURM_JOB_ID} ${output_dir}"
  fi

  info "Done at $(date +%Y-%m-%d.%H:%M:%S)"

  # Cleanup trap signals and force kill on same signal to propagate
  trap - ${signal}
  trap - EXIT
  if [[ "${signal}" != "SIGUSR1" ]]; then
    kill -${signal} $$
  fi
}

trap 'cleanupOnExit "SIGHUP"' SIGHUP
trap 'cleanupOnExit "SIGINT"' SIGINT
trap 'cleanupOnExit "SIGTERM"' SIGTERM
trap 'cleanupOnExit "SIGQUIT"' SIGQUIT
trap 'cleanupOnExit "SIGUSR1"' SIGUSR1
trap 'cleanupOnExit "SIGUSR1"' USR1
trap 'cleanupOnExit "EXIT"' EXIT


######################################################################
# Launch pythons simulation
######################################################################
export PYTHONUNBUFFERED=1
simulation.py