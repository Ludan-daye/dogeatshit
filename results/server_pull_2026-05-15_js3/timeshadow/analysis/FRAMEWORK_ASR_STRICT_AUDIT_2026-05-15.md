# Framework ASR Strict Audit (2026-05-15)

Scope: deterministic triage of old regex-era framework raw logs under
`server_pull_2026-04-20`. This is not a replacement for an LLM judge. It is a
pre-judge audit to identify obvious cases where `clean=true` conflicts with
the response text.

Script: `analysis/framework_strict_audit.py`

Output: `analysis/framework_strict_audit_2026-05-15.json`

## Main Finding

The draft's framework ASR figures (LangGraph 80%, CrewAI 40%, AutoGen 23%) are
not safe to present as strict attack rates. They are legacy `ASR_clean` labels
from script/regex-style classifiers. Several `clean=true` trials contain
explicit refusals, and several others are only task mentions or execution
claims without a produced artifact.

## Condition-Level Summary

| Raw file | Condition | Old clean count | Strict triage counts | Old clean but refusal | Old clean ambiguous/other |
|---|---:|---:|---|---:|---:|
| M1 framework | LG_R2_attack | 24/30 | ATTACK_CLAIM 18, REFUSE 8, AMBIG_TASK 2, AMBIG_EXEC 1, ACK 1 | 3 | 3 |
| M17 cron | LG17_R2_attack_cron | 24/30 | ATTACK_CLAIM 16, REFUSE 6, AMBIG_EXEC 5, AMBIG_TASK 2, ACK 1 | 2 | 7 |
| M1 framework | CA_R2_attack | 12/30 | REFUSE 23, ATTACK_CLAIM 4, AMBIG_EXEC 1, OTHER 2 | 7 | 1 |
| M11 CrewAI | CA_R2_real_attack_task | 12/30 | ATTACK_CLAIM 15, AMBIG_EXEC 9, AMBIG_TASK 6 | 0 | 8 |
| M18 AutoGen | AG18_R2_attack_agent_hist | 7/30 | ACK 21, REFUSE 6, ATTACK_CLAIM 2, AMBIG_TASK 1 | 4 | 1 |

## Paper Implication

- Keep the statement that the attack reaches all three frameworks only if it is
  phrased as "framework delivery / compliance-claim under the controlled
  refusal boundary".
- Do not call 80/40/23 "strict judge-verified ATTACK" until a real LLM-judge
  pass over the full responses is run.
- Report strict categories separately: REFUSE, ACK, ATTACK_CLAIM, AMBIG.
- CrewAI and AutoGen are the most affected by this correction.

## Next Step

Run the LLM judge with the v2 strict prompt over these same framework responses
and write a new `judge_framework_strict_*.json`. This should be done before the
next paper draft uses framework ASR as a main claim.
