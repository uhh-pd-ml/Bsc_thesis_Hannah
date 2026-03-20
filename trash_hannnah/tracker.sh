#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <command> [args...]"
  exit 1
fi

logfile="/beegfs/u/bbd1146/memlog.$(date +%Y%m%d_%H%M%S).txt"

# Start the target command in background
"$@" &
pid=$!

echo "# Logging memory of PID $pid" | tee -a "$logfile"
echo "# Command: $*" | tee -a "$logfile"
echo "# Time                 Elapsed_s   VmRSS_kB   VmHWM_kB   VmSize_kB   %MEM" | tee -a "$logfile"

start_epoch=$(date +%s.%N)

while kill -0 "$pid" 2>/dev/null; do
  now_human=$(date '+%F %T')
  now_epoch=$(date +%s.%N)
  elapsed=$(awk "BEGIN {print $now_epoch - $start_epoch}")

  if [ -r "/proc/$pid/status" ]; then
    vmrss=$(awk '/VmRSS:/  {print $2}' "/proc/$pid/status")
    vmhwm=$(awk '/VmHWM:/  {print $2}' "/proc/$pid/status")
    vmsize=$(awk '/VmSize:/ {print $2}' "/proc/$pid/status")
  else
    vmrss="NA"
    vmhwm="NA"
    vmsize="NA"
  fi

  pmem=$(ps -p "$pid" -o %mem= 2>/dev/null | awk '{print $1}')
  pmem=${pmem:-NA}

  printf "%-20s %-11s %-10s %-10s %-11s %-6s\n" \
    "$now_human" "$elapsed" "$vmrss" "$vmhwm" "$vmsize" "$pmem" \
    | tee -a "$logfile"

  sleep 0.2
done

wait "$pid"
rc=$?

echo "# Process exited with code $rc" | tee -a "$logfile"
exit "$rc"