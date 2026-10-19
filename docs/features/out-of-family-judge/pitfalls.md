# Out-of-family RT-DETR judge — Pitfalls

- [ ] [OUT-OF-FAMILY-JUDGE-05] Under pytest, `scripts/campaign_report.py`'s `logging.basicConfig` is a no-op (pytest already put a handler on the root logger), so the tables `t2o aggregate` prints in section 6 never reach `capsys`; read them from `caplog.messages` after `caplog.set_level(logging.INFO)` — otherwise a test asserting on them silently passes on the echoed `$ t2o aggregate --metric ...` line or fails for a reason that looks like missing rows
