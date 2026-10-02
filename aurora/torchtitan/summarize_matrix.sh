#!/bin/bash
# Summarise the 4-node configuration matrix: one row per arm.
cd "$(dirname "$0")"
printf "%-16s %5s %7s %9s %9s %9s %9s %9s\n" ARM STEPS GRADNORM FWD_BWD BATCH GEN_PULL TOKS_FBW TOKS_FULL
for L in $(ls -t runlogs/*_4n.log 2>/dev/null); do
  tag=$(grep -aoE "matrix_[a-z0-9_]+" "$L" | head -1 | sed 's/matrix_//')
  [ -z "$tag" ] && continue
  steps=$(grep -aoE "Train \| Step: *[0-9]+" "$L" | awk '{print $NF}' | uniq | tail -1)
  med () { grep -aoE "$1: [0-9.e+-]+" "$L" | awk '{print $2}' | uniq | sort -g | awk '{a[NR]=$1} END{if(NR)printf "%.3g", a[int((NR+1)/2)]}'; }
  printf "%-16s %5s %7s %9s %9s %9s %9s %9s\n" "$tag" "${steps:-0}" \
    "$(med 'trainer/grad_norm/mean')" \
    "$(med 'perf/trainer/step_time_ratio/fwd_bwd')" \
    "$(med 'perf/trainer/step_time_ratio/batch')" \
    "$(med 'perf/trainer/step_time_ratio/blocking_generator_pull_model_state_dict')" \
    "$(med 'perf/trainer/tokens_per_second_fwd_bwd')" \
    "$(med 'perf/trainer/tokens_per_second_full_step')"
done
