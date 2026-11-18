# More than a Hack 2027 — Phase 1: Ball Possession

## 1. Overview

This project addresses the ball-possession classification task from **More than a Hack 2027 — Phase 1: Ball Possession**.

Given one complete FC Barcelona vs Athletic Club match with automatic tracking, tactical tagging, and homography data, the objective is to assign one of four labels to every video frame:

- `H` — FC Barcelona in possession
- `A` — Athletic Club in possession
- `D` — dead ball
- `X` — no answer / abstention

The final possession file contains one prediction for each of the **199,319 video frames**.

Because full-match possession ground truth was not available to participants, the pipeline uses **algorithmically generated pseudo-labels** derived from the provided match assets. No human annotations of the provided match and no private ground truth were used for model training or validation.

---

## 2. Approach

The pipeline is divided into four stages.

### 2.1 Data loading and alignment

The first stage loads and validates the provided match assets:

- Automatic Tracking Data (ATD)
- Tactical homography
- Broadcast homography
- Smart Tagging
- Tactical video metadata

The ATD contains **199,319 consecutive frames**, aligned one-to-one with the tactical video.

The tactical homography contains valid mappings for all 199,319 frames. Broadcast homography was treated as a separate asset because it does not share the same complete tactical timeline.

Smart Tagging was used as an additional temporal signal, particularly for identifying match-state events such as kickoffs.

The data-loading stage also verifies frame continuity, coordinate conventions, pitch dimensions, and frame/time alignment before feature generation.

---

## 3. Feature Engineering

The tracking data was transformed into frame-level tactical features.

### 3.1 Distance to ball

For each tracked player, the Euclidean distance to the ball was calculated after converting normalized tracking coordinates into pitch-space metres.

This provides a direct measure of how close each player is to the ball.

### 3.2 Player velocity

Player speed values supplied by the tracking data were extracted as player-level speed features.

### 3.3 Derived ball velocity

The original ball-speed field was unavailable for the complete sequence.

Therefore, ball speed was reconstructed algorithmically from consecutive valid ball coordinates:

1. calculate the displacement between consecutive valid ball positions;
2. convert normalized coordinates to pitch-space metres;
3. divide by the elapsed frame time;
4. express the resulting speed in metres per second.

The reconstruction uses the supplied 25 FPS tactical timeline.

A rolling version of ball speed was also experimentally evaluated, but it was not adopted in the final model.

### 3.4 Ball visibility

A binary `ball_visible` feature was created from the availability of both ball coordinates.

### 3.5 Nearest player team

For every frame with a valid ball position, the nearest tracked player was identified.

The corresponding team was encoded as:

- `0` — Barcelona (`H`)
- `1` — Athletic Club (`A`)

The player-slot/team mapping was verified against the repository's existing possession pipeline.

---

## 4. Algorithmic Pseudo-Label Generation

A full-match human-labelled possession dataset was not available and manual annotation of the provided match was not permitted.

Instead, possession labels were generated algorithmically using signals already present in the supplied data.

The pseudo-label generation combines:

- Smart Tagging;
- dead-ball information;
- ATD tracking information;
- nearest-player team;
- temporal majority-vote smoothing.

The resulting labels are therefore **algorithm-generated pseudo-labels**, not ground truth.

A 50-frame temporal majority vote was used to reduce frame-level label jitter.

The final pseudo-label distribution after smoothing was:

| Label | Count |
|---|---:|
| `H` | 46,614 |
| `A` | 36,381 |
| `D` | 57,032 |
| `X` | 59,292 |

These values describe the generated pseudo-labels and should not be interpreted as the true possession distribution of the match.

---

## 5. Model Training

A Random Forest classifier was used for the live-ball possession classification.

Dead-ball frames were handled separately using the dead-ball signal rather than requiring the classifier to infer them from player features.

The model configuration used for the hybrid pipeline was:

- Random Forest
- 200 trees
- `random_state=42`
- `max_features="sqrt"`
- `n_jobs=-1`

The chronological nature of football data was preserved during the train/test split.

The first **80% of the match frames** were used for model development and the remaining **20%** were held out chronologically.

The trained model was then used to generate predictions across the complete match.

Because the first 80% is used for model training, predictions on that portion are in-sample predictions. The final 20% provides the chronological out-of-sample portion of the model evaluation against pseudo-labels.

No private competition ground truth was used during this process.

---

## 6. Feature Experiment

An experiment compared the feature set before and after adding the `nearest_player_team` feature.

Using the algorithmically generated pseudo-labels as the evaluation target:

| Feature set | Accuracy | Macro F1 |
|---|---:|---:|
| 88 features | 0.9152 | 0.8059 |
| 89 features | 0.9760 | 0.9459 |

The additional feature substantially increased agreement with the generated pseudo-labels.

However, this result **does not represent true possession accuracy**, because the target labels were themselves generated algorithmically.

Further analysis showed that `nearest_player_team` matched the pseudo-label team signal on **113,441 of 113,488 ball-present frames (99.9586%)**.

This indicates strong coupling between the feature and the pseudo-label generation process. Therefore, the large pseudo-label reproduction score should not be interpreted as evidence of equivalent performance against unseen human-labelled ground truth.

---

## 7. Public Evaluation

The repository's official evaluation pipeline was used to validate and evaluate the generated submissions.

The public ground truth contains only a limited portion of the match and is intended for pipeline checking rather than model training or tuning.

The evaluation covered:

- **7,500 graded frames**
- **17 ground-truth possession spells**
- **21 ground-truth segments**

Three locally generated candidates were structurally validated:

- baseline
- hybrid
- V4

### Public diagnostic results

| Candidate | Accuracy | Macro F1 | Spell F1 Mean |
|---|---:|---:|---:|
| Baseline | 0.6476 | 0.6355 | 0.2612 |
| Hybrid | 0.6476 | 0.6355 | 0.2612 |
| V4 | 0.6771 | 0.6630 | 0.3282 |

The V4 candidate produced the strongest public diagnostic results among the tested candidates.

Its additional diagnostic values included:

- Home F1: `0.7981`
- Away F1: `0.6891`
- Dead-ball F1: `0.5017`
- Accuracy: `0.6771`
- Macro F1: `0.6630`
- Spell F1 mean: `0.3282`
- Transition F1: `0.1395`

V4 also produced fewer predicted possession spells and lower fragmentation than the baseline/hybrid candidates.

However, its transition F1 was lower than the baseline/hybrid candidates, indicating that possession-boundary detection remained a weakness.

---

## 8. Interpretation of the Public Results

The public results should be treated strictly as **diagnostic evidence**.

The public ground truth is too limited to establish full-match performance, and it was not used for tuning or candidate selection.

Bootstrap analysis using 1,000 iterations produced overlapping confidence intervals between candidates. Therefore, the observed improvement of V4 over the baseline should not be described as statistically conclusive.

The results nevertheless provide useful evidence about the behaviour of the pipeline:

1. The system can generate structurally valid full-match predictions.
2. The model performs better on the public diagnostic window than the initial baseline.
3. Possession boundaries and transitions remain difficult.
4. Pseudo-label reproduction performance is substantially higher than public-ground-truth performance.
5. The gap between pseudo-label evaluation and public evaluation is an important limitation of the approach.

---

## 9. Possession Rules Implemented

The pipeline distinguishes four output states:

### Home possession — `H`

Assigned when the algorithmic signals indicate Barcelona as the possessing team.

### Away possession — `A`

Assigned when the algorithmic signals indicate Athletic Club as the possessing team.

### Dead ball — `D`

Dead-ball states are identified using the algorithmic match-state/dead-ball signals available in the supplied data and pseudo-label pipeline.

### Abstention — `X`

`X` is used for frames where the pipeline does not establish an active possession state, including match-state periods such as non-play intervals.

No manual frame-by-frame corrections were applied.

---

## 10. Training Data

No external human-labelled possession dataset was used.

The provided FC Barcelona vs Athletic Club match was **not manually annotated**.

The model-development target was generated algorithmically from the supplied match assets using the pseudo-label pipeline described above.

The public five-minute ground truth was not used for training, pseudo-label generation, model tuning, or candidate ranking.

No private ground truth was accessed.

---

## 11. Limitations

### 11.1 Pseudo-label dependence

The largest limitation is that the training targets are algorithmically generated pseudo-labels rather than complete human-labelled possession ground truth.

This means the model can learn to reproduce the assumptions of the pseudo-label generator rather than necessarily learning the true possession state.

### 11.2 Feature-target coupling

`nearest_player_team` is extremely closely related to the pseudo-label generation process.

Its 99.9586% agreement with the pseudo-label team signal on ball-present frames demonstrates that the feature experiment measures pseudo-label reproducibility more than independent possession understanding.

### 11.3 Ball visibility

The tracking data contains substantial periods where the ball is not visible.

This limits the reliability of direct ball-relative features and makes possession inference dependent on surrounding player and temporal information.

### 11.4 Transition detection

The public evaluation indicates that transition detection remains weak.

The final V4 candidate achieved a transition F1 of approximately `0.140`, showing that correctly identifying possession changes is substantially harder than maintaining an existing possession state.

### 11.5 Limited public ground truth

Only a small public evaluation window is available.

Consequently, public metrics cannot reliably estimate performance across the entire private evaluation set.

### 11.6 Generalisation gap

The pseudo-label reproduction performance was substantially higher than the public evaluation performance.

This demonstrates that the model's agreement with its generated training targets should not be treated as a direct measure of competition performance.

---

## 12. Reproduction

The repository contains the source code required to reproduce the pipeline when the participant has access to the competition data bundle.

The general pipeline is:

```bash
python starter/load_atd.py
python starter/load_homography.py
python starter/load_smart_tagging.py
```

Feature generation and model preparation are implemented under:

```text
features/
models/
```

The evaluation pipeline can be executed with:

```bash
python evaluation/run_evaluation.py
```

A generated possession file should be structurally validated before submission:

```bash
python evaluation/validate_submission.py <submission.csv>
```

The expected submission format is:

```text
frame,label
0,X
1,X
...
199318,X
```

The repository does not contain the private match data or full-match submission files.

---

## 13. Final Submission State

The final repository contains the source code required for the team's possession pipeline while keeping the supplied match data and full-match derived outputs outside version control.

Full-match submission files and evaluation artifacts remain local and ignored by Git.

The final candidate selected for submission was **V4**, based on its strongest observed public diagnostic performance among the tested candidates.

This selection should be understood as an engineering decision based on the available diagnostic evidence, not as proof of superior performance on the private ground truth.

---

## 14. Conclusion

The project developed an end-to-end possession prediction pipeline from automatic football tracking data, tactical tagging, and pitch geometry.

The central challenge was the absence of full-match public possession ground truth. To address this, the pipeline used algorithmically generated pseudo-labels and tactical features, followed by Random Forest classification and temporal processing.

The final experiments demonstrated a clear distinction between **pseudo-label reproduction** and **actual possession prediction**. While the model achieved high agreement with its generated pseudo-labels, public evaluation performance was considerably lower.

This gap is the main limitation of the current approach and provides the clearest direction for future improvement: obtaining stronger independent possession supervision, improving possession-boundary detection, and reducing dependence between the target-generation mechanism and the model features.