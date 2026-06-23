# Experiment summary

- runs collected: **64**
- distinct cells: **10**
- distinct meetings: **8**

## Meeting difficulty (sorted by overlap_ratio desc)

| meeting_id | duration_s | num_speakers | num_turns | turns_per_min | overlap_s | overlap_ratio | mean_turn_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R8007_M8010 | 1856.322 | 4 | 1181 | 38.172 | 1046.740 | 0.564 | 2.754 |
| R8001_M8004 | 1573.850 | 4 | 604 | 23.026 | 456.340 | 0.290 | 3.393 |
| R8007_M8011 | 1858.980 | 4 | 768 | 24.788 | 442.780 | 0.238 | 3.029 |
| R8003_M8001 | 2068.000 | 4 | 674 | 19.555 | 333.500 | 0.161 | 3.362 |
| R8008_M8013 | 2235.490 | 3 | 732 | 19.647 | 310.040 | 0.139 | 3.211 |
| R8009_M8019 | 1964.660 | 2 | 959 | 29.288 | 187.340 | 0.095 | 1.990 |
| R8009_M8018 | 1651.480 | 2 | 665 | 24.160 | 116.300 | 0.070 | 2.422 |
| R8009_M8020 | 1907.600 | 2 | 874 | 27.490 | 125.300 | 0.066 | 2.139 |

## Per-(cell × meeting) scores

| cell_id | meeting_id | cer | wer | spk_acc | routing_f1 | overlap_f1 | events | llm_resolved | wall_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| asr=faster-whisper_osd=energy_fallback_resolver=none_sep=none | R8001_M8004_MS801 | 0.618 | None | 0.637 | 0.607 | 0.501 | 342 | 0.000 | 163.118 |
| asr=faster-whisper_osd=energy_fallback_resolver=none_sep=none | R8003_M8001_MS801 | 0.395 | None | 0.707 | 0.540 | 0.517 | 413 | 0.000 | 187.527 |
| asr=faster-whisper_osd=energy_fallback_resolver=none_sep=none | R8007_M8010_MS803 | 0.697 | None | 0.504 | 0.688 | 0.702 | 381 | 0.000 | 236.803 |
| asr=faster-whisper_osd=energy_fallback_resolver=none_sep=none | R8007_M8011_MS806 | 0.473 | None | 0.654 | 0.626 | 0.565 | 350 | 0.000 | 201.822 |
| asr=faster-whisper_osd=energy_fallback_resolver=none_sep=none | R8008_M8013_MS807 | 0.610 | None | 0.593 | 0.400 | 0.319 | 538 | 0.000 | 177.587 |
| asr=faster-whisper_osd=energy_fallback_resolver=none_sep=none | R8009_M8018_MS809 | 0.448 | None | 0.811 | 0.462 | 0.294 | 355 | 0.000 | 172.079 |
| asr=faster-whisper_osd=energy_fallback_resolver=none_sep=none | R8009_M8019_MS810 | 0.363 | None | 0.893 | 0.241 | 0.147 | 382 | 0.000 | 127.291 |
| asr=faster-whisper_osd=energy_fallback_resolver=none_sep=none | R8009_M8020_MS810 | 0.208 | None | 0.860 | 0.139 | 0.102 | 221 | 0.000 | 113.350 |
| asr=faster-whisper_osd=energy_fallback_resolver=none_sep=sepformer | R8001_M8004_MS801 | 0.620 | None | 0.646 | 0.607 | 0.501 | 342 | 0.000 | 162.863 |
| asr=faster-whisper_osd=energy_fallback_resolver=none_sep=sepformer | R8003_M8001_MS801 | 0.395 | None | 0.705 | 0.540 | 0.517 | 413 | 0.000 | 188.625 |
| asr=faster-whisper_osd=energy_fallback_resolver=none_sep=sepformer | R8007_M8010_MS803 | 0.686 | None | 0.509 | 0.688 | 0.702 | 381 | 0.000 | 233.039 |
| asr=faster-whisper_osd=energy_fallback_resolver=none_sep=sepformer | R8007_M8011_MS806 | 0.467 | None | 0.649 | 0.626 | 0.565 | 350 | 0.000 | 200.976 |
| asr=faster-whisper_osd=energy_fallback_resolver=none_sep=sepformer | R8008_M8013_MS807 | 0.600 | None | 0.593 | 0.400 | 0.319 | 538 | 0.000 | 177.034 |
| asr=faster-whisper_osd=energy_fallback_resolver=none_sep=sepformer | R8009_M8018_MS809 | 0.460 | None | 0.814 | 0.462 | 0.294 | 355 | 0.000 | 173.235 |
| asr=faster-whisper_osd=energy_fallback_resolver=none_sep=sepformer | R8009_M8019_MS810 | 0.339 | None | 0.893 | 0.241 | 0.147 | 382 | 0.000 | 127.533 |
| asr=faster-whisper_osd=energy_fallback_resolver=none_sep=sepformer | R8009_M8020_MS810 | 0.208 | None | 0.855 | 0.139 | 0.102 | 221 | 0.000 | 113.796 |
| asr=faster-whisper_osd=energy_fallback_resolver=openai_sep=none | R8001_M8004_MS801 | 0.616 | None | 0.594 | 0.607 | 0.501 | 47 | 0.257 | 1325.436 |
| asr=faster-whisper_osd=energy_fallback_resolver=openai_sep=none | R8003_M8001_MS801 | 0.394 | None | 0.683 | 0.540 | 0.517 | 45 | 0.254 | 1191.382 |
| asr=faster-whisper_osd=energy_fallback_resolver=openai_sep=none | R8007_M8010_MS803 | 0.683 | None | 0.449 | 0.688 | 0.702 | 51 | 0.339 | 1930.445 |
| asr=faster-whisper_osd=energy_fallback_resolver=openai_sep=none | R8007_M8011_MS806 | 0.429 | None | 0.563 | 0.626 | 0.565 | 44 | 0.343 | 1480.642 |
| asr=faster-whisper_osd=energy_fallback_resolver=openai_sep=none | R8008_M8013_MS807 | 0.587 | None | 0.571 | 0.400 | 0.319 | 50 | 0.091 | 1224.773 |
| asr=faster-whisper_osd=energy_fallback_resolver=openai_sep=none | R8009_M8018_MS809 | 0.401 | None | 0.786 | 0.462 | 0.294 | 43 | 0.287 | 1238.414 |
| asr=faster-whisper_osd=energy_fallback_resolver=openai_sep=none | R8009_M8019_MS810 | 0.340 | None | 0.901 | 0.241 | 0.147 | 382 | 0.013 | 172.541 |
| asr=faster-whisper_osd=energy_fallback_resolver=openai_sep=none | R8009_M8020_MS810 | 0.208 | None | 0.860 | 0.139 | 0.102 | 221 | 0.000 | 119.752 |
| asr=faster-whisper_osd=energy_fallback_resolver=openai_sep=sepformer | R8001_M8004_MS801 | 0.617 | None | 0.649 | 0.607 | 0.501 | 342 | 0.000 | 220.080 |
| asr=faster-whisper_osd=energy_fallback_resolver=openai_sep=sepformer | R8003_M8001_MS801 | 0.395 | None | 0.709 | 0.540 | 0.517 | 413 | 0.000 | 252.764 |
| asr=faster-whisper_osd=energy_fallback_resolver=openai_sep=sepformer | R8007_M8010_MS803 | 0.693 | None | 0.509 | 0.688 | 0.702 | 381 | 0.000 | 320.838 |
| asr=faster-whisper_osd=energy_fallback_resolver=openai_sep=sepformer | R8007_M8011_MS806 | 0.473 | None | 0.651 | 0.626 | 0.565 | 350 | 0.000 | 267.533 |
| asr=faster-whisper_osd=energy_fallback_resolver=openai_sep=sepformer | R8008_M8013_MS807 | 0.643 | None | 0.602 | 0.400 | 0.319 | 538 | 0.000 | 230.713 |
| asr=faster-whisper_osd=energy_fallback_resolver=openai_sep=sepformer | R8009_M8019_MS810 | 0.337 | None | 0.890 | 0.241 | 0.147 | 27 | 0.039 | 508.386 |
| asr=faster-whisper_osd=energy_fallback_resolver=openai_sep=sepformer | R8009_M8020_MS810 | 0.207 | None | 0.846 | 0.139 | 0.102 | 17 | 0.041 | 427.191 |
| asr=faster-whisper_osd=pyannote_resolver=none_sep=none | R8001_M8004_MS801 | 0.613 | None | 0.646 | 0.607 | 0.501 | 342 | 0.000 | 166.472 |
| asr=faster-whisper_osd=pyannote_resolver=none_sep=none | R8003_M8001_MS801 | 0.433 | None | 0.707 | 0.540 | 0.517 | 413 | 0.000 | 191.250 |
| asr=faster-whisper_osd=pyannote_resolver=none_sep=none | R8007_M8010_MS803 | 0.686 | None | 0.512 | 0.688 | 0.702 | 381 | 0.000 | 237.261 |
| asr=faster-whisper_osd=pyannote_resolver=none_sep=none | R8007_M8011_MS806 | 0.451 | None | 0.643 | 0.626 | 0.565 | 350 | 0.000 | 202.164 |
| asr=faster-whisper_osd=pyannote_resolver=none_sep=none | R8008_M8013_MS807 | 0.613 | None | 0.599 | 0.400 | 0.319 | 538 | 0.000 | 178.465 |
| asr=faster-whisper_osd=pyannote_resolver=none_sep=none | R8009_M8018_MS809 | 0.456 | None | 0.811 | 0.462 | 0.294 | 355 | 0.000 | 172.968 |
| asr=faster-whisper_osd=pyannote_resolver=none_sep=none | R8009_M8019_MS810 | 0.340 | None | 0.893 | 0.241 | 0.147 | 382 | 0.000 | 128.413 |
| asr=faster-whisper_osd=pyannote_resolver=none_sep=none | R8009_M8020_MS810 | 0.208 | None | 0.860 | 0.139 | 0.102 | 221 | 0.000 | 113.970 |
| asr=faster-whisper_osd=pyannote_resolver=none_sep=sepformer | R8001_M8004_MS801 | 0.628 | None | 0.649 | 0.607 | 0.501 | 342 | 0.000 | 167.284 |
| asr=faster-whisper_osd=pyannote_resolver=none_sep=sepformer | R8003_M8001_MS801 | 0.396 | None | 0.709 | 0.540 | 0.517 | 413 | 0.000 | 387.172 |
| asr=faster-whisper_osd=pyannote_resolver=none_sep=sepformer | R8007_M8010_MS803 | 0.707 | None | 0.507 | 0.688 | 0.702 | 381 | 0.000 | 480.218 |
| asr=faster-whisper_osd=pyannote_resolver=none_sep=sepformer | R8007_M8011_MS806 | 0.457 | None | 0.651 | 0.626 | 0.565 | 350 | 0.000 | 407.365 |
| asr=faster-whisper_osd=pyannote_resolver=none_sep=sepformer | R8009_M8018_MS809 | 0.446 | None | 0.814 | 0.462 | 0.294 | 355 | 0.000 | 355.687 |
| asr=faster-whisper_osd=pyannote_resolver=none_sep=sepformer | R8009_M8019_MS810 | 0.339 | None | 0.893 | 0.241 | 0.147 | 382 | 0.000 | 244.854 |
| asr=faster-whisper_osd=pyannote_resolver=none_sep=sepformer | R8009_M8020_MS810 | 0.208 | None | 0.860 | 0.139 | 0.102 | 221 | 0.000 | 111.788 |
| asr=faster-whisper_osd=pyannote_resolver=openai_sep=none | R8001_M8004_MS801 | 0.618 | None | 0.596 | 0.607 | 0.501 | 47 | 0.240 | 1535.047 |
| asr=faster-whisper_osd=pyannote_resolver=openai_sep=none | R8007_M8010_MS803 | 0.693 | None | 0.457 | 0.688 | 0.702 | 52 | 0.304 | 2010.567 |
| asr=faster-whisper_osd=pyannote_resolver=openai_sep=none | R8007_M8011_MS806 | 0.427 | None | 0.560 | 0.626 | 0.565 | 49 | 0.326 | 1766.198 |
| asr=faster-whisper_osd=pyannote_resolver=openai_sep=none | R8009_M8018_MS809 | 0.401 | None | 0.761 | 0.462 | 0.294 | 51 | 0.296 | 1580.277 |
| asr=faster-whisper_osd=pyannote_resolver=openai_sep=none | R8009_M8020_MS810 | 0.207 | None | 0.851 | 0.139 | 0.102 | 13 | 0.027 | 403.694 |
| asr=faster-whisper_osd=pyannote_resolver=openai_sep=sepformer | R8001_M8004_MS801 | 0.610 | None | 0.582 | 0.607 | 0.501 | 48 | 0.234 | 1355.692 |
| asr=faster-whisper_osd=pyannote_resolver=openai_sep=sepformer | R8007_M8010_MS803 | 0.672 | None | 0.459 | 0.688 | 0.702 | 44 | 0.344 | 2154.923 |
| asr=faster-whisper_osd=pyannote_resolver=openai_sep=sepformer | R8007_M8011_MS806 | 0.443 | None | 0.560 | 0.626 | 0.565 | 39 | 0.311 | 1475.632 |
| asr=faster-whisper_osd=pyannote_resolver=openai_sep=sepformer | R8008_M8013_MS807 | 0.597 | None | 0.559 | 0.400 | 0.319 | 41 | 0.097 | 1077.507 |
| asr=faster-whisper_osd=pyannote_resolver=openai_sep=sepformer | R8009_M8018_MS809 | 0.387 | None | 0.758 | 0.462 | 0.294 | 52 | 0.293 | 1294.631 |
| asr=faster-whisper_osd=pyannote_resolver=openai_sep=sepformer | R8009_M8019_MS810 | 0.340 | None | 0.893 | 0.241 | 0.147 | 23 | 0.037 | 360.738 |
| asr=faster-whisper_osd=pyannote_resolver=openai_sep=sepformer | R8009_M8020_MS810 | 0.207 | None | 0.855 | 0.139 | 0.102 | 14 | 0.041 | 327.240 |
| asr=funasr_osd=pyannote_resolver=none_sep=none | R8001_M8004_MS801 | 0.508 | None | 0.529 | 0.607 | 0.501 | 344 | 0.000 | 168.139 |
| asr=funasr_osd=pyannote_resolver=none_sep=none | R8003_M8001_MS801 | 0.298 | None | 0.620 | 0.540 | 0.517 | 413 | 0.000 | 192.292 |
| asr=funasr_osd=pyannote_resolver=none_sep=none | R8007_M8010_MS803 | 0.698 | None | 0.476 | 0.688 | 0.702 | 380 | 0.000 | 192.678 |
| asr=funasr_osd=pyannote_resolver=none_sep=none | R8007_M8011_MS806 | 0.458 | None | 0.566 | 0.626 | 0.565 | 350 | 0.000 | 309.686 |
| asr=funasr_osd=pyannote_resolver=none_sep=sepformer | R8009_M8019_MS810 | 0.200 | None | 0.895 | 0.241 | 0.147 | 382 | 0.000 | 202.833 |
| asr=funasr_osd=pyannote_resolver=none_sep=sepformer | R8009_M8020_MS810 | 0.095 | None | 0.842 | 0.139 | 0.102 | 221 | 0.000 | 192.672 |
