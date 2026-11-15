# 038 — FLIR seed check: ultralytics' own per-epoch files confirm the three 0 px label-transfer seeds trained identically, so record 036's identical seeds are not a log artifact

**Date:** 2026-11-15 · **Task:** REGISTRATION-AXIS-06 · **Machine:** not recorded · **Wall clock:** not recorded
**git_sha:** not recorded · **W&B:** none · **Log:** `logs/2026-11-15-REGISTRATION-AXIS-06-seed-check.txt`

Record 036 found that FLIR's three label-transfer seeds gave identical results (F227). The evidence
there was the sweep's log, condensed on the server before it was pasted back, and a hand-copied
CSV. Either could have introduced the identity. This check reads the files ultralytics writes for
each run on the server, which went through neither step. It trained nothing. It re-reads three of
record 036's runs, the 0 px cells at seeds 0, 1 and 2.

## Command

Run by hand in PowerShell, in
`D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\detectors`:

```
foreach ($s in 0, 1, 2) {
    $rows = Import-Csv "flir-l0-s$s\results.csv"
    $keep = $rows[0].PSObject.Properties.Name | Where-Object { $_.Trim() -ne 'time' }
    $txt  = ($rows | Select-Object -Property $keep | ConvertTo-Csv -NoTypeInformation) -join "`n"
    $hash = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash(
                [Text.Encoding]::UTF8.GetBytes($txt))) -replace '-', ''
    "s$s  rows $($rows.Count)  hash-without-time $hash"
}
Select-String -Path flir-l0-s*\args.yaml -Pattern '^(seed|deterministic):' |
    ForEach-Object { "$(Split-Path (Split-Path $_.Path) -Leaf)  $($_.Line)" }
Get-Content flir-l0-s0\results.csv -TotalCount 3
Get-Content flir-l0-s1\results.csv -TotalCount 3
```

## Environment

```
not recorded
```

## Configuration

The three runs are record 036's `flir-l0-s0`, `flir-l0-s1` and `flir-l0-s2`: `yolo11s.pt`, 100
epochs, all 4,129 FLIR train images at 0 px misalignment (tree digest `45aab9344c36a4f0`). The hash
covers every column of ultralytics' `results.csv` except `time`, the epoch's wall-clock seconds,
which differs between any two runs.

## Results

Verbatim from the log, after its echoed command:

```
s0  rows 100  hash-without-time 7F38F444D923617026A6CE4441E9C2B76C4B7204C680F51D65CB7A89C262B6A7
s1  rows 100  hash-without-time 7F38F444D923617026A6CE4441E9C2B76C4B7204C680F51D65CB7A89C262B6A7
s2  rows 100  hash-without-time 7F38F444D923617026A6CE4441E9C2B76C4B7204C680F51D65CB7A89C262B6A7
flir-l0-s0  seed: 0
flir-l0-s0  deterministic: true
flir-l0-s1  seed: 1
flir-l0-s1  deterministic: true
flir-l0-s2  seed: 2
flir-l0-s2  deterministic: true
epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2
1,59.2669,1.74059,1.15729,1.20064,0.68766,0.40143,0.43568,0.20419,1.74298,1.18435,1.30088,0.000415058,0.000415058,0.000415058
2,124.33,1.71106,1.09074,1.19559,0.41153,0.40726,0.38952,0.16836,1.87212,1.4264,1.37743,0.000823491,0.000823491,0.000823491
epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2
1,57.3153,1.74059,1.15729,1.20064,0.68766,0.40143,0.43568,0.20419,1.74298,1.18435,1.30088,0.000415058,0.000415058,0.000415058
2,111.783,1.71106,1.09074,1.19559,0.41153,0.40726,0.38952,0.16836,1.87212,1.4264,1.37743,0.000823491,0.000823491,0.000823491
```

An earlier attempt hashed the whole files, `time` included. That gave three different hashes
(`C0CE0112…`, `C4657E42…`, `17AAD122…`), as expected from the differing epoch times. It printed
no readable seed lines and was not saved as a log.

## Findings

### F233 — FLIR's seed reached ultralytics and changed nothing: the three 0 px runs carry `seed: 0`, `1` and `2` in `args.yaml`, yet their `results.csv` files agree in every column but `time` for all 100 epochs

All three files hash to `7F38F444…62B6A7` once `time` is removed. In the two files shown, epochs 1
and 2 match in every loss, precision, recall, mAP and learning-rate value. Epoch 1 of seed 0 reads
0.43568 mAP50 and 1.74059 box loss, and so does seed 1. Only the epoch times differ (59.27 s against
57.32 s), so these are separate runs, not copies. This confirms F227 from files that never passed
through the log condensing or the hand-copied CSV, and it rules both out as the cause. It also
shows the seed was not lost before ultralytics: each run's own config records the seed it was
given, with `deterministic: true`. **Why ultralytics ignores the seed on FLIR is still not
established.** This check does not test it.

## Provenance caveats

- **No provenance block.** The commands were typed into a PowerShell window on the server and the
  output pasted into the session. Machine, git SHA and environment are not recorded. The path in
  the prompt is the server's repo checkout, and it reads files written by record 036's runs.
- **Only the 0 px level was checked.** The other four levels rest on record 036's log comparison,
  which showed the same identity at all five.
- **Only two files' first two epochs are printed.** Identity over all 100 epochs and all three
  seeds rests on the hash, which covers every row and every column except `time`.
- **The log was saved from the session's paste**, verbatim, and not by the server's `Start-Step`
  tee.

## Next

Find the cause before any FLIR detector campaign with more than one seed. Record 036's Next section
already proposes a cheap test: two short `yolo11s.pt` runs on FLIR at seeds 0 and 1, compared at the
first epoch.
