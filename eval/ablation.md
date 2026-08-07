# Component Ablation

Each configuration is scored over the **same 15 held-out rows** from `sample_messages.csv` (the last 15 by `message_id`; see `eval/split.json`). Held-out rows never enter any prompt in any configuration, and the 15 exemplars are re-rendered under each ablation so the prompt stays internally consistent.

Decision model: `claude-sonnet-4-6`, temperature 0, structured tool output.

| # | configuration | action | message_type | driver | both a+t | rows citing evidence |
|---|---|---|---|---|---|---|
| a | rubric only | **11/15** | **11/15** | 4/15 | 9/15 | 0/15 |
| b | + context features | **14/15** | **13/15** | 8/15 | 13/15 | 0/15 |
| c | + retrieval & evidence | **14/15** | **13/15** | 8/15 | 13/15 | 13/15 |
| d | + media enrichment | **15/15** | **15/15** | 9/15 | 15/15 | 13/15 |
| e | full (as shipped) | **15/15** | **15/15** | 10/15 | 15/15 | 13/15 |
| f | full + fast path (retired) | **15/15** | **15/15** | 9/15 | 15/15 | 13/15 |

## What each configuration adds

- **a — rubric only**: Message text + rubric. No context features, no retrieval, no media.
- **b — + context features**: Adds user / group / business / relationship / temporal / deadline / template.
- **c — + retrieval & evidence**: Adds the candidate pool, near-duplicate aggregates and evidence citation.
- **d — + media enrichment**: Adds OCR text, ASR transcripts and the extracted media facts.
- **e — full (as shipped)**: Adds the deterministic credential/payment solicitation scan.
- **f — full + fast path (retired)**: As shipped, but with ENABLE_FAST_PATH=True. The retired configuration.

## Per-row detail

### a — rubric only

| message_id | action | type | driver |
|---|---|---|---|
| sample_msg_019 | OK mute/mute | OK scam/scam | OK |
| sample_msg_020 | OK mute/mute | OK scam/scam | - |
| sample_msg_041 | OK digest/digest | **MISS** unknown/personal | - |
| sample_msg_042 | **MISS** digest/notify | **MISS** unknown/urgent | - |
| sample_msg_043 | **MISS** digest/mute | **MISS** unknown/spam | - |
| sample_msg_044 | OK digest/digest | OK promotion/promotion | - |
| sample_msg_045 | **MISS** digest/mute | OK promotion/promotion | - |
| sample_msg_046 | OK notify/notify | OK event/event | OK |
| sample_msg_047 | **MISS** digest/mute | OK promotion/promotion | - |
| sample_msg_048 | OK digest/digest | OK business_update/business_update | - |
| sample_msg_049 | OK digest/digest | OK unknown/unknown | OK |
| sample_msg_050 | OK digest/digest | OK personal/personal | - |
| sample_msg_051 | OK notify/notify | OK urgent/urgent | - |
| sample_msg_052 | OK mute/mute | OK scam/scam | - |
| sample_msg_053 | OK mute/mute | **MISS** spam/scam | OK |

### b — + context features

| message_id | action | type | driver |
|---|---|---|---|
| sample_msg_019 | OK mute/mute | OK scam/scam | OK |
| sample_msg_020 | OK mute/mute | OK scam/scam | OK |
| sample_msg_041 | OK digest/digest | **MISS** unknown/personal | - |
| sample_msg_042 | **MISS** digest/notify | **MISS** unknown/urgent | - |
| sample_msg_043 | OK mute/mute | OK spam/spam | OK |
| sample_msg_044 | OK digest/digest | OK promotion/promotion | - |
| sample_msg_045 | OK mute/mute | OK promotion/promotion | - |
| sample_msg_046 | OK notify/notify | OK event/event | OK |
| sample_msg_047 | OK mute/mute | OK promotion/promotion | OK |
| sample_msg_048 | OK digest/digest | OK business_update/business_update | - |
| sample_msg_049 | OK digest/digest | OK unknown/unknown | OK |
| sample_msg_050 | OK digest/digest | OK personal/personal | - |
| sample_msg_051 | OK notify/notify | OK urgent/urgent | OK |
| sample_msg_052 | OK mute/mute | OK scam/scam | - |
| sample_msg_053 | OK mute/mute | OK scam/scam | OK |

### c — + retrieval & evidence

| message_id | action | type | driver |
|---|---|---|---|
| sample_msg_019 | OK mute/mute | OK scam/scam | OK |
| sample_msg_020 | OK mute/mute | OK scam/scam | - |
| sample_msg_041 | OK digest/digest | **MISS** unknown/personal | - |
| sample_msg_042 | **MISS** digest/notify | **MISS** unknown/urgent | - |
| sample_msg_043 | OK mute/mute | OK spam/spam | OK |
| sample_msg_044 | OK digest/digest | OK promotion/promotion | - |
| sample_msg_045 | OK mute/mute | OK promotion/promotion | OK |
| sample_msg_046 | OK notify/notify | OK event/event | OK |
| sample_msg_047 | OK mute/mute | OK promotion/promotion | OK |
| sample_msg_048 | OK digest/digest | OK business_update/business_update | - |
| sample_msg_049 | OK digest/digest | OK unknown/unknown | OK |
| sample_msg_050 | OK digest/digest | OK personal/personal | - |
| sample_msg_051 | OK notify/notify | OK urgent/urgent | OK |
| sample_msg_052 | OK mute/mute | OK scam/scam | - |
| sample_msg_053 | OK mute/mute | OK scam/scam | OK |

### d — + media enrichment

| message_id | action | type | driver |
|---|---|---|---|
| sample_msg_019 | OK mute/mute | OK scam/scam | OK |
| sample_msg_020 | OK mute/mute | OK scam/scam | - |
| sample_msg_041 | OK digest/digest | OK personal/personal | - |
| sample_msg_042 | OK notify/notify | OK urgent/urgent | OK |
| sample_msg_043 | OK mute/mute | OK spam/spam | OK |
| sample_msg_044 | OK digest/digest | OK promotion/promotion | - |
| sample_msg_045 | OK mute/mute | OK promotion/promotion | OK |
| sample_msg_046 | OK notify/notify | OK event/event | OK |
| sample_msg_047 | OK mute/mute | OK promotion/promotion | OK |
| sample_msg_048 | OK digest/digest | OK business_update/business_update | - |
| sample_msg_049 | OK digest/digest | OK unknown/unknown | OK |
| sample_msg_050 | OK digest/digest | OK personal/personal | - |
| sample_msg_051 | OK notify/notify | OK urgent/urgent | OK |
| sample_msg_052 | OK mute/mute | OK scam/scam | - |
| sample_msg_053 | OK mute/mute | OK scam/scam | OK |

### e — full (as shipped)

| message_id | action | type | driver |
|---|---|---|---|
| sample_msg_019 | OK mute/mute | OK scam/scam | OK |
| sample_msg_020 | OK mute/mute | OK scam/scam | OK |
| sample_msg_041 | OK digest/digest | OK personal/personal | - |
| sample_msg_042 | OK notify/notify | OK urgent/urgent | OK |
| sample_msg_043 | OK mute/mute | OK spam/spam | OK |
| sample_msg_044 | OK digest/digest | OK promotion/promotion | - |
| sample_msg_045 | OK mute/mute | OK promotion/promotion | OK |
| sample_msg_046 | OK notify/notify | OK event/event | OK |
| sample_msg_047 | OK mute/mute | OK promotion/promotion | OK |
| sample_msg_048 | OK digest/digest | OK business_update/business_update | - |
| sample_msg_049 | OK digest/digest | OK unknown/unknown | OK |
| sample_msg_050 | OK digest/digest | OK personal/personal | - |
| sample_msg_051 | OK notify/notify | OK urgent/urgent | OK |
| sample_msg_052 | OK mute/mute | OK scam/scam | - |
| sample_msg_053 | OK mute/mute | OK scam/scam | OK |

### f — full + fast path (retired)

| message_id | action | type | driver |
|---|---|---|---|
| sample_msg_019 | OK mute/mute | OK scam/scam | OK |
| sample_msg_020 | OK mute/mute | OK scam/scam | OK |
| sample_msg_041 | OK digest/digest | OK personal/personal | - |
| sample_msg_042 | OK notify/notify | OK urgent/urgent | OK |
| sample_msg_043 | OK mute/mute | OK spam/spam | OK |
| sample_msg_044 | OK digest/digest | OK promotion/promotion | - |
| sample_msg_045 | OK mute/mute | OK promotion/promotion | OK |
| sample_msg_046 | OK notify/notify | OK event/event | OK |
| sample_msg_047 | OK mute/mute | OK promotion/promotion | - |
| sample_msg_048 | OK digest/digest | OK business_update/business_update | - |
| sample_msg_049 | OK digest/digest | OK unknown/unknown | OK |
| sample_msg_050 | OK digest/digest | OK personal/personal | - |
| sample_msg_051 | OK notify/notify | OK urgent/urgent | OK |
| sample_msg_052 | OK mute/mute | OK scam/scam | - |
| sample_msg_053 | OK mute/mute | OK scam/scam | OK |
