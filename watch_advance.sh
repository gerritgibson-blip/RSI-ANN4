#!/bin/bash
# exits (printing the event) when a new ADVANCE event appears, or when the trn service stops running
EV=/workspace/trn_state/events.jsonl
n0=$(grep -c '"ADVANCE"' $EV 2>/dev/null || true); n0=${n0:-0}
while true; do
  n=$(grep -c "\"ADVANCE\"" $EV 2>/dev/null || true); n=${n:-0}
  if [ "$n" -gt "$n0" ]; then grep '"ADVANCE"' $EV | tail -n $((n-n0)); tail -3 $EV | grep -v ADVANCE; exit 0; fi
  if ! supervisorctl status trn | grep -q RUNNING; then echo "SERVICE NOT RUNNING"; tail -30 /var/log/portal/trn.log; exit 1; fi
  sleep 20
done
