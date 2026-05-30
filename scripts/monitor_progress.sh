#!/bin/bash
# monitor_progress.sh — 实时监控 scheduler.ts 进度（只读文件系统，不修改任何运行中进程）
# Usage: watch -n 5 -c /data3/paper_analysis/scripts/monitor_progress.sh <work_dir>
#    or: while true; do clear; /data3/paper_analysis/scripts/monitor_progress.sh <work_dir>; sleep 5; done

WORK_DIR="${1:-/data3/paper_analysis/learning_outputs}"

LAYER_ORDER=(L1 L2 L3 L4 L5 L6)
LAYER_NAMES=("算法Pipeline" "Serving调度" "编译框架" "Kernel调度" "硬件架构" "芯片设计")
COLOR_RESET="\033[0m"
COLOR_GREEN="\033[32m"
COLOR_YELLOW="\033[33m"
COLOR_CYAN="\033[36m"
COLOR_RED="\033[31m"
COLOR_BOLD="\033[1m"
COLOR_DIM="\033[2m"
COLOR_BLUE="\033[34m"

# ── Phase detection ──
phase1_done=0; phase2_done=0; phase3_done=0; phase4_done=0
phase1_count=0; phase2_count=0; phase3_count=0; phase4_count=0
total_questions=0

# ── Phase 1: Question Spaces ──
for i in "${!LAYER_ORDER[@]}"; do
  lid="${LAYER_ORDER[$i]}"
  qf="${WORK_DIR}/${lid}_问题空间.md"
  if [ -f "$qf" ] && grep -q "\[QUESTION_AGENT_DONE\]" "$qf" 2>/dev/null; then
    ((phase1_count++))
    # Count questions in this layer
    qcnt=$(grep -oP 'Q\d\.\d+' "$qf" 2>/dev/null | sort -u | wc -l)
    total_questions=$((total_questions + qcnt))
  fi
done
if [ $phase1_count -eq 6 ]; then phase1_done=1; fi

# ── Phase 2: Answers ──
# Find all expected question IDs from question spaces
declare -A answer_status
declare -A answer_size
if [ $phase1_done -eq 1 ]; then
  for i in "${!LAYER_ORDER[@]}"; do
    lid="${LAYER_ORDER[$i]}"
    qf="${WORK_DIR}/${lid}_问题空间.md"
    if [ -f "$qf" ]; then
      while IFS= read -r qid; do
        af="${WORK_DIR}/${qid}_${lid}_answer.md"
        if [ -f "$af" ] && grep -q "\[ANSWER_AGENT_DONE\]" "$af" 2>/dev/null; then
          answer_status["${qid}_${lid}"]="done"
          answer_size["${qid}_${lid}"]=$(wc -c < "$af" 2>/dev/null)
          ((phase2_count++))
        elif [ -f "$af" ]; then
          answer_status["${qid}_${lid}"]="writing"
        else
          answer_status["${qid}_${lid}"]="pending"
        fi
      done < <(grep -oP 'Q\d\.\d+' "$qf" 2>/dev/null | sort -u)
    fi
  done
fi

# Check running processes
running_answer=$(pgrep -f "answer_Q[0-9]" 2>/dev/null | wc -l)
running_horizon=$(pgrep -f "horizon_" 2>/dev/null | wc -l)
running_vertical=$(pgrep -f "vertical_summary" 2>/dev/null | wc -l)

# ── Phase 3: Horizon ──
for i in "${!LAYER_ORDER[@]}"; do
  lid="${LAYER_ORDER[$i]}"
  hf="${WORK_DIR}/${lid}_horizon_summary.md"
  if [ -f "$hf" ] && grep -q "\[HORIZON_SUMMARY_DONE\]" "$hf" 2>/dev/null; then
    ((phase3_count++))
  fi
done
if [ $phase3_count -eq 6 ]; then phase3_done=1; fi

# ── Phase 4: Vertical ──
vf="${WORK_DIR}/summary.md"
if [ -f "$vf" ] && grep -q "\[VERTICAL_SUMMARY_DONE\]" "$vf" 2>/dev/null; then
  phase4_done=1
fi

# ── Determine current phase ──
if [ $phase4_done -eq 1 ]; then current_phase="4 (完成!)"; current_color=$COLOR_GREEN
elif [ $phase3_done -eq 1 ] || [ $running_horizon -gt 0 ] || [ $running_vertical -gt 0 ]; then current_phase="3/4 (Horizon/Vertical)"; current_color=$COLOR_CYAN
elif [ $phase1_done -eq 1 ]; then current_phase="2 (Answer)"; current_color=$COLOR_YELLOW
else current_phase="1 (Question)"; current_color=$COLOR_BLUE
fi

# ── Elapsed ──
dispatch="${WORK_DIR}/dispatch.json"
if [ -f "$dispatch" ]; then
  start_ts=$(stat -c %Y "$dispatch" 2>/dev/null)
  now_ts=$(date +%s)
  elapsed=$((now_ts - start_ts))
  elapsed_str="$(($elapsed / 3600))h $(($elapsed % 3600 / 60))m $(($elapsed % 60))s"
else
  elapsed_str="?"
fi

# ── Render ──
echo -e "${COLOR_BOLD}╔══════════════════════════════════════════════════════════════╗${COLOR_RESET}"
echo -e "${COLOR_BOLD}║${COLOR_RESET}  Scheduler Progress Monitor   ${COLOR_DIM}elapsed: ${elapsed_str}${COLOR_RESET}"
echo -e "${COLOR_BOLD}╠══════════════════════════════════════════════════════════════╣${COLOR_RESET}"

# Phase 1 bar
p1_bar=""
for i in $(seq 1 6); do
  if [ $i -le $phase1_count ]; then p1_bar+="${COLOR_GREEN}█${COLOR_RESET}"
  else p1_bar+="${COLOR_DIM}░${COLOR_RESET}"; fi
done
p1_status="${COLOR_GREEN}DONE${COLOR_RESET}"
echo -e "${COLOR_BOLD}║${COLOR_RESET} Phase 1 Question:  [${p1_bar}] ${phase1_count}/6  ${p1_status}"

# Phase 2 bar
p2_total=35  # known from previous run
p2_pct=$((phase2_count * 100 / p2_total))
p2_bar=""
p2_blocks=$((phase2_count * 30 / p2_total))
for i in $(seq 1 30); do
  if [ $i -le $p2_blocks ]; then p2_bar+="${COLOR_YELLOW}█${COLOR_RESET}"
  else p2_bar+="${COLOR_DIM}░${COLOR_RESET}"; fi
done
if [ $phase2_count -eq $p2_total ]; then p2_color=$COLOR_GREEN; p2_label="DONE"; else p2_color=$COLOR_YELLOW; p2_label="RUNNING"; fi
printf "${COLOR_BOLD}║${COLOR_RESET} Phase 2 Answer:    [%b] %2d/%-2d %3d%%  %b\n" "$p2_bar" "$phase2_count" "$p2_total" "$p2_pct" "${p2_color}${p2_label} (${running_answer} active)${COLOR_RESET}"

# Phase 3 bar
p3_bar=""
for i in $(seq 1 6); do
  if [ $i -le $phase3_count ]; then p3_bar+="${COLOR_CYAN}█${COLOR_RESET}"
  else p3_bar+="${COLOR_DIM}░${COLOR_RESET}"; fi
done
if [ $phase3_done -eq 1 ]; then p3_color=$COLOR_GREEN; p3_label="DONE";
elif [ $phase3_count -gt 0 ] || [ $running_horizon -gt 0 ]; then p3_color=$COLOR_CYAN; p3_label="RUNNING";
else p3_color=$COLOR_DIM; p3_label="WAITING"; fi
echo -e "${COLOR_BOLD}║${COLOR_RESET} Phase 3 Horizon:   [${p3_bar}] ${phase3_count}/6  ${p3_color}${p3_label}${COLOR_RESET}"

# Phase 4 bar
if [ $phase4_done -eq 1 ]; then p4_color=$COLOR_GREEN; p4_label="DONE ✓";
elif [ -f "$vf" ] || [ $running_vertical -gt 0 ]; then p4_color=$COLOR_CYAN; p4_label="RUNNING";
else p4_color=$COLOR_DIM; p4_label="WAITING"; fi
echo -e "${COLOR_BOLD}║${COLOR_RESET} Phase 4 Vertical:  ${p4_color}${p4_label}${COLOR_RESET}"

echo -e "${COLOR_BOLD}╠══════════════════════════════════════════════════════════════╣${COLOR_RESET}"

# ── Active Workers ──
echo -e "${COLOR_BOLD}║${COLOR_RESET} ${COLOR_DIM}Active:${COLOR_RESET} answer=${COLOR_YELLOW}${running_answer}${COLOR_RESET} horizon=${COLOR_CYAN}${running_horizon}${COLOR_RESET} vertical=${COLOR_BLUE}${running_vertical}${COLOR_RESET}"

# ── Per-layer detail ──
if [ $phase1_done -eq 1 ] && [ $phase2_done -eq 0 ]; then
  echo -e "${COLOR_BOLD}╠══════════════════════════════════════════════════════════════╣${COLOR_RESET}"
  for i in "${!LAYER_ORDER[@]}"; do
    lid="${LAYER_ORDER[$i]}"
    lname="${LAYER_NAMES[$i]}"
    # Count done / total for this layer
    ldone=0; ltotal=0
    for key in "${!answer_status[@]}"; do
      if [[ "$key" == *"_${lid}" ]]; then
        ((ltotal++))
        if [ "${answer_status[$key]}" = "done" ]; then ((ldone++)); fi
      fi
    done
    if [ $ltotal -eq 0 ]; then continue; fi
    # Build mini bar
    lbar=""
    for j in $(seq 1 $ltotal); do
      if [ $j -le $ldone ]; then lbar+="${COLOR_GREEN}●${COLOR_RESET}"
      else lbar+="${COLOR_DIM}○${COLOR_RESET}"; fi
    done
    lstatus=""
    if [ $ldone -eq $ltotal ]; then lstatus="${COLOR_GREEN}✓${COLOR_RESET}"; fi
    printf "${COLOR_BOLD}║${COLOR_RESET} %-14s [%s] %d/%d %s\n" "${lid} ${lname}" "$lbar" "$ldone" "$ltotal" "$lstatus"
  done
fi

echo -e "${COLOR_BOLD}╚══════════════════════════════════════════════════════════════╝${COLOR_RESET}"
echo -e "${COLOR_DIM}refresh: $(date '+%H:%M:%S') | work_dir: ${WORK_DIR}${COLOR_RESET}"
