#!/bin/bash
HEXP=/sda/zgh/stage1_5_experiment; LOG=$HEXP/preds/p0_full.log; WLOG=$HEXP/preds/p0_watchdog.log
[ -f /tmp/p0_pause ] && { echo "[$(date '+%H:%M:%S')] paused, skip" >> $WLOG; exit 0; }
exec 9>/tmp/p0_watchdog.lock; flock -n 9 || { echo "[$(date '+%H:%M:%S')] 已有看门狗, skip" >> $WLOG; exit 0; }
echo "[$(date '+%m-%d %H:%M:%S')] tick" >> $WLOG
if grep -q "P0_FULL_INFER_DONE" $LOG 2>/dev/null; then
  [ -s $HEXP/eval/P0_FULL_RESULTS.txt ] || { echo "  done->scoring" >> $WLOG; python3 $HEXP/scripts/score_bootstrap.py full > $HEXP/eval/P0_FULL_RESULTS.txt 2>&1; echo "  scored" >> $WLOG; }
  exit 0
fi
if pgrep -f "p0_full_messidor.sh" >/dev/null 2>&1; then echo "  running ok" >> $WLOG
else echo "  relaunch" >> $WLOG; setsid nohup bash $HEXP/scripts/p0_full_messidor.sh >> $HEXP/preds/p0_full_outer.log 2>&1 < /dev/null & sleep 5; fi
